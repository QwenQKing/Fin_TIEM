from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path
from openai import OpenAI
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it=None, **k):
        return it if it is not None else []
import foresight.config as cfg
from foresight.retry import with_retry
from foresight.stores.sample_store import SampleStore
from foresight.stores.text_kg import TextKG
from foresight.stores.experience import ExperienceLibrary
from foresight.stores.cache import Cache
BASE = Path(__file__).parent.parent
DBROOT = BASE / 'databases'
CASE_LOG = BASE / 'case_log'
CASE_LOG.mkdir(exist_ok=True)

def kb_dir(dataset: str) -> Path:
    return DBROOT / dataset

def resolve_kb(name: str | None) -> Path:
    if name:
        return DBROOT / name
    subs = [d for d in DBROOT.iterdir() if d.is_dir()] if DBROOT.exists() else []
    if len(subs) == 1:
        return subs[0]
    names = [d.name for d in subs]
    raise SystemExit(f'databases/ has {len(subs)} datasets, please specify --kb <name>: {names}')

def retrieved_path(db: Path, dataset: str) -> Path:
    d = db / 'retrieved'
    d.mkdir(parents=True, exist_ok=True)
    return d / f'{dataset}.json'
from foresight.llm import chat_client
_client = chat_client()
import threading
_client_lock = threading.Lock()

def _rebuild_client():
    global _client
    with _client_lock:
        _client = chat_client()
    return _client

def _chat(messages, max_tokens):
    _temp = float(os.environ.get('LLM_TEMPERATURE', '0'))
    try:
        r = _client.chat.completions.create(model=cfg.LLM_MODEL, temperature=_temp, max_tokens=max_tokens, messages=messages)
    except Exception as e:
        if '401' in str(e) or type(e).__name__ == 'AuthenticationError':
            r = _rebuild_client().chat.completions.create(model=cfg.LLM_MODEL, temperature=_temp, max_tokens=max_tokens, messages=messages)
        else:
            raise
    if not getattr(r, 'choices', None):
        msg = getattr(r, 'status_msg', None) or getattr(r, 'error', None) or 'response has no choices'
        raise RuntimeError(f'LLM return choices( token /rate-limit): {msg}')
    return r
TOPIC = {'earnings': 'quarterly/annual earnings', 'earnings_preann': 'earnings pre-announcement & orders', 'ma': 'M&A progress', 'regulatory': 'regulatory compliance & risk', 'index_change': 'index membership & liquidity', 'rating': 'analyst ratings & reports', 'guidance': 'full-year guidance & outlook', 'large_move': 'material price-moving event'}
POS_OUTCOMES = {'', '', 'done', '', '', '', ''}

def _arg_val(flag: str, default=None):
    import sys
    a = sys.argv
    if flag in a and a.index(flag) + 1 < len(a):
        return a[a.index(flag) + 1]
    return default

def load_catalysts(catalysts_path: str | None, ss, db: Path) -> tuple:
    split = _arg_val('--split')
    if catalysts_path:
        p = Path(catalysts_path)
        if not p.is_absolute():
            p = BASE / p
        ej = json.loads(p.read_text(encoding='utf-8'))
        evs = ej.get('catalysts', [])
        ds = ej.get('meta', {}).get('dataset', 'eval')
    else:
        ds = json.loads((db / '01_info' / 'meta.json').read_text(encoding='utf-8')).get('dataset', 'dataset')
        evs = list(ss.catalysts)
    if split:
        evs = [e for e in evs if e.get('split') == split]
        ds = f'{ds}-{split}'
    miss = sorted({e['stock_id'] for e in evs} - set(ss.stocks))
    if miss:
        print(f'⚠ {len(miss)} eventsstocks KB({db.name}) →  OOD( event_text + ): {miss[:6]}')
    print(f"predict {len(evs)} events (split={split or ''}); KB={db.name}")
    return (evs, ds)

def direction(outcome) -> int:
    if isinstance(outcome, (int, float)):
        return 1 if outcome > 0 else -1
    o = str(outcome).strip().lower()
    if o in ('positive', 'pos', 'up', '1'):
        return 1
    if o in ('negative', 'neg', 'down', '-1', '0'):
        return -1
    return 1 if outcome in POS_OUTCOMES else -1

def case_to_txt(c: dict) -> str:
    (m, I, O) = (c['meta'], c['INPUT'], c['OUTPUT'])
    L = []
    L += ['========== TASK ==========', f"predict {m['stock_name']}({m['stock_id']}) [{m['catalyst_type']}] event: decision {m['decision_time']} result.", '']
    L += ['========== RETRIEVE ==========']
    L += ['[event] ' + (I.get('event_text') or '(none)')[:300]]
    L += ['[news facts]'] + ([f"  [{x['ts']}] {x['fact']}" for x in I['news_facts']] or ['  (none)'])
    L += ['[experiences]'] + ([f"  [adv {e.get('advantage', 0):+.2f}] {e.get('name', '')}: {e.get('description', '')}" for e in I['experiences']] or ['  (none)'])
    L += ['', '========== LLM PROMPT ==========', I['prompt'], '']
    L += ['========== LLM RESPONSE ==========', O['raw_response'], '']
    (g, e) = (c['GROUND_TRUTH'], c['EVAL'])
    L += ['========== EVAL ==========', f"truth: {g['outcome']} ({g['true_direction']})", f"predict: {O['pred_direction']} (conf={O.get('pred_conf_verbal')})", f"result: {('correct' if e['correct'] else 'wrong')}"]
    return '\n'.join(L)

def _safe_model(m: str) -> str:
    return (m or 'model').replace('/', '_').replace(':', '_')

def case_root(dataset: str, model: str) -> Path:
    run_id = os.environ.get('RUN_ID', '').strip()
    suffix = f'-run{run_id}' if run_id else ''
    return CASE_LOG / _safe_model(model) / f'{dataset}{suffix}'

def save_case(case: dict, dataset: str, model: str):
    d = case_root(dataset, model) / case['catalyst_id']
    d.mkdir(parents=True, exist_ok=True)
    (d / 'case.json').write_text(json.dumps(case, ensure_ascii=False, indent=1), encoding='utf-8')
    (d / 'log.txt').write_text(case_to_txt(case), encoding='utf-8')

def _market_of(stock_id: str) -> str:
    s = (stock_id or '').upper()
    if s.endswith('.US'):
        return 'U.S. equity'
    if s.endswith(('.SZ', '.SH', '.BJ')):
        return 'A-share (China)'
    return 'equity'

def _adaptive_budget(streams: dict, total_budget_chars: int=8000, priorities: tuple=('focal', 'exp', 'episode', 'theme', 'prior')) -> dict:
    base_shares = {'focal': 0.2, 'exp': 0.3, 'episode': 0.18, 'theme': 0.12, 'prior': 0.2}
    initial = {n: int(total_budget_chars * base_shares[n]) for n in priorities}
    take = {n: 0 for n in priorities}
    surplus = 0
    for n in priorities:
        budget_n = initial[n] + surplus
        actual = min(len(streams.get(n, '')), budget_n)
        take[n] = actual
        surplus = budget_n - actual
    for n in priorities:
        if surplus <= 0:
            break
        s = streams.get(n, '')
        if take[n] < len(s):
            extra = min(surplus, len(s) - take[n])
            take[n] += extra
            surplus -= extra
    out = {}
    for n in priorities:
        s = streams.get(n, '')
        if len(s) <= take[n]:
            out[n] = s
        else:
            out[n] = s[:take[n]] + ' ...[truncated]'
    return out

def build_prompt(name, etype, T, event_text, facts, exps, concurrent=None, episodes=None, themes=None, market='A-share (China)', horizon=5, total_budget_chars: int=8000) -> str:
    focal_raw = (event_text or '').strip() or '(event content unavailable)'
    prior_raw = '\n'.join((f"  - [{f['ts']}] ({f.get('stock')}) {f['fact']}" for f in facts)) or '  (no prior evidence)'
    exp_raw = '\n'.join((f"  - [adv {e.get('advantage', 0):+.2f}] {e.get('name', '')}: {e.get('description', '')}" for e in exps)) or '  (no prior experience)'
    ep_lines = []
    for e in episodes or []:
        ts_s = (e.get('tau_start') or '')[:10] or '?'
        ts_e = (e.get('tau_end') or '')[:10] or '?'
        ep_lines.append(f"  - [{e.get('type', 'other')} | {ts_s}→{ts_e}] {e.get('summary', '')}")
    episode_raw = '\n'.join(ep_lines)
    th_lines = []
    for t in themes or []:
        ts_s = (t.get('tau_start') or '')[:10] or '?'
        ts_e = (t.get('tau_end') or '')[:10] or '?'
        th_lines.append(f"  - [{t.get('type', 'other')} | {ts_s}→{ts_e}] {t.get('name', '')}: {t.get('description', '')}")
    theme_raw = '\n'.join(th_lines)
    streams = {'focal': focal_raw, 'prior': prior_raw, 'exp': exp_raw, 'episode': episode_raw, 'theme': theme_raw}
    cut = _adaptive_budget(streams, total_budget_chars=total_budget_chars)
    (ev_txt, fc, ex, ep_str, th_str) = (cut['focal'], cut['prior'], cut['exp'], cut['episode'], cut['theme'])
    cc = '\n'.join((f"  - [{c['ts']}] {c['fact']}" for c in concurrent or []))
    cc_block = f'\n[Concurrent events on/near the decision day — consider their JOINT configuration]\n{cc}' if cc else ''
    ep_block = '\n[Recent episodes — multi-day causal chains on this stock, all ended BEFORE decision time]\n' + ep_str if ep_str else ''
    th_block = '\n[Related cross-stock themes — sector/policy waves, all closed BEFORE decision time]\n' + th_str if th_str else ''
    return f"""You are an expert {market} analyst forecasting the SHORT-TERM MARKET REACTION to a corporate event. Given the event below and ONLY pre-event evidence, judge whether the company's stock RETURN over the NEXT {horizon} TRADING DAYS will be POSITIVE (up) or NEGATIVE (down).\n(POSITIVE = the stock rises over the next {horizon} trading days; NEGATIVE = it falls. Weigh BOTH the event's fundamentals AND the likely market reaction — strong news may already be priced in, weak news may be over-sold.)\n\n[Catalyst] Company: {name}; Catalyst type: {etype} ({TOPIC.get(etype, '')}); Decision time: {T}\n[The event / announcement to react to]\n{ev_txt}{cc_block}{ep_block}{th_block}\n[Prior event evidence (disclosures/news) - ALL dated BEFORE the decision time]\n{fc}\n[Similar historical experience (transferable lessons)]\n{ex}\n\nFIRST write your full reasoning inside <think></think> tags (reason from the event's fundamentals, prior evidence, experience, and likely market reaction), THEN output JSON:\n<think>your step-by-step reasoning over the event content, prior evidence, and experience</think>\n{{"prob_positive": float in 0-1 (probability the {horizon}-day return is positive), "direction": "positive" or "negative", "reasoning": "brief justification"}}"""

def retrieve_evidence(ev, ss, tk, el, cache) -> dict:
    (s, T, et) = (ev['stock_id'], ev['decision_time'], ev['catalyst_type'])
    name = ss.name(s)
    industry = ss.stocks.get(s, {}).get('industry_l1')
    focal = tk.retrieve_focal(s, T, k=cfg.FOCAL_K, pit_strict=not cfg.PIT_OFF)
    event_text = focal.get('text') or ''
    focal_facts = focal.get('facts', [])
    query = f"{TOPIC.get(et, '')} {name}"
    ckey = cache.key(query, T, ['textkg', str(cfg.CONTEXT_K), 'samestock'])
    facts = cache.get(ckey)
    if facts is None:
        facts = tk.query(query, top_k=cfg.CONTEXT_K, before_ts=T, stock=s)
        cache.set(ckey, facts)
    focal_ts = {focal.get('ts')} | {ff.get('ts') for ff in focal_facts}
    focal_ts.discard(None)
    facts = [f for f in facts if not (f.get('stock') == s and f.get('ts') in focal_ts)]
    exp_query = f"{TOPIC.get(et, '')}. {(event_text or '')[:400]}"
    exps = el.retrieve(
        et, exp_query, k=cfg.EXP_K, before_date=T,
        min_sim=cfg.EXP_MIN_SIM, domain_boost=cfg.EXP_DOMAIN_BOOST,
        use_stability_weight=cfg.USE_CSM,
        strict_provenance=os.getenv('CSM_STRICT_PROVENANCE', '1') == '1')
    concurrent = tk.concurrent_events(s, T)
    episodes = themes = None
    if cfg.USE_C2F:
        c2f = tk.retrieve_c2f(stock=s, T=T, query=f"{TOPIC.get(et, '')} {name} {(event_text or '')[:200]}")
        episodes = c2f.get('l2') or []
        themes = c2f.get('l3') or []
    return {'name': name, 'industry': industry, 'et': et, 'T': T, 'market': _market_of(s), 'horizon': int(ev.get('horizon_days', 5) or 5), 'event_text': event_text, 'focal_facts': focal_facts, 'facts': facts, 'exps': exps, 'concurrent': concurrent, 'episodes': episodes, 'themes': themes}

def predict_catalyst(ev, ss, tk, el, cache, evidence=None, use_scmr: bool=None, episodes: list=None, themes: list=None, vote_uniform: bool=False) -> dict:
    if use_scmr is None:
        use_scmr = cfg.USE_SCMR
    E = evidence if evidence is not None else retrieve_evidence(ev, ss, tk, el, cache)
    (name, et, T, industry) = (E['name'], E['et'], E['T'], E['industry'])
    (event_text, facts, exps) = (E['event_text'], E['facts'], E['exps'])
    focal_facts = E.get('focal_facts', [])
    concurrent = E.get('concurrent', [])
    if episodes is None:
        episodes = E.get('episodes')
    if themes is None:
        themes = E.get('themes')
    if use_scmr and len(exps) >= 1:
        try:
            from foresight.scmr import SCMRunner
            runner = SCMRunner(tk, el, cache, vote_uniform=vote_uniform)
            focal = {'text': event_text, 'facts': focal_facts}
            scmr_out = runner.predict(ev, x_event=event_text, exps=exps, focal_evidence=focal)
            if scmr_out is not None:
                trail = scmr_out['trail']
                best_path = max(trail, key=lambda p: p['w_i']) if trail else {'rationale_i': ''}
                think_all = '\n---\n'.join((p['rationale_i'] for p in trail))
                return {'name': name, 'industry': industry, 'catalyst_type': et, 'INPUT': {'event_text': event_text, 'focal_facts': focal_facts, 'news_facts': facts, 'experiences': exps, 'concurrent': concurrent, 'episodes': episodes or [], 'themes': themes or [], 'v6_flags': {'USE_C2F': cfg.USE_C2F, 'USE_CSM': cfg.USE_CSM, 'USE_SCMR': cfg.USE_SCMR, 'PIT_OFF': cfg.PIT_OFF}, 'prompt': '(scmr fork-reason-converge, no single prompt)'}, 'OUTPUT': {'raw_response': json.dumps(trail, ensure_ascii=False), 'think': think_all, 'pred_conf_verbal': scmr_out['conf'], 'pred_direction': scmr_out['pred_direction'], 'reasoning': best_path['rationale_i'], 'model': cfg.LLM_MODEL, 'usage': {}, 'trail': trail, 'n_scmr_paths': scmr_out['n_paths']}}
        except Exception as e:
            print(f'  ⚠ SCMR fail, fallback to single path: {type(e).__name__}: {e}', file=sys.stderr)
    prompt = build_prompt(name, et, T, event_text, facts, exps, concurrent=concurrent, episodes=episodes, themes=themes, market=E.get('market', 'A-share (China)'), horizon=E.get('horizon', 5))
    _ckey_run = os.environ.get('RUN_ID', '').strip()
    _ckey_extra = f'\x00scmr={use_scmr}\x00llm={cfg.LLM_MODEL}\x00run={_ckey_run}'
    pkey = cache.key(prompt + _ckey_extra, '', ['llm'])
    cached = cache.get(pkey)
    if cached is None:
        r = with_retry(lambda : _chat([{'role': 'user', 'content': prompt}], cfg.PREDICT_MAX_TOKENS), max_retry=cfg.LLM_MAX_RETRY, backoff=cfg.LLM_RETRY_BACKOFF, what='predict')
        cached = {'raw': r.choices[0].message.content, 'usage': {'prompt_tokens': r.usage.prompt_tokens, 'completion_tokens': r.usage.completion_tokens}}
        cache.set(pkey, cached)
    raw = cached['raw']
    import re
    think_m = re.search('<think>(.*?)</think>', raw, re.S)
    think = think_m.group(1).strip() if think_m else ''
    json_m = re.search('\\{[^{}]*?\\"prob_positive\\".*?\\}', raw, re.S)
    try:
        o = json.loads(json_m.group(0) if json_m else raw)
        conf = float(max(0.0, min(1.0, o.get('prob_positive', 0.5))))
        reason = o.get('reasoning', '')
    except Exception:
        (conf, reason) = (0.5, '(parse failed)')
    pdir = 'positive' if conf >= 0.5 else 'negative'
    return {'name': name, 'industry': industry, 'catalyst_type': et, 'INPUT': {'event_text': event_text, 'focal_facts': focal_facts, 'news_facts': facts, 'experiences': exps, 'concurrent': concurrent, 'episodes': episodes or [], 'themes': themes or [], 'v6_flags': {'USE_C2F': cfg.USE_C2F, 'USE_CSM': cfg.USE_CSM, 'USE_SCMR': cfg.USE_SCMR, 'PIT_OFF': cfg.PIT_OFF}, 'prompt': prompt}, 'OUTPUT': {'raw_response': raw, 'think': think, 'pred_conf_verbal': conf, 'pred_direction': pdir, 'reasoning': reason, 'model': cfg.LLM_MODEL, 'usage': cached.get('usage', {})}}
_DISTILL_DOMAINS = 'earnings/earnings_preann/ma/regulatory/index_change/rating/guidance/large_move'

def acquire_skill(ev, res, correct: bool, cache) -> list:
    O = res['OUTPUT']
    horizon = int(ev.get('horizon_days', ev.get('horizon', 5)) or 5)
    reasoning = (O.get('reasoning') or O.get('think') or '')[:600]
    verdict = 'CORRECT' if correct else 'WRONG'
    guide = "Extract 1-2 generalizable lessons about WHY the reasoning succeeded — which heuristic / evidence source (the event's fundamentals, prior evidence, experience) was decisive." if correct else 'Extract 1-2 generalizable lessons about WHAT went wrong — what bias / missing info / reasoning error led to the miss, and what to do differently next time.'
    prompt = f"""You are analyzing an A-share forecast of the {horizon}-day post-event stock-return direction, to extract a REUSABLE CAUSAL SKILL — a transferable heuristic linking event features to expected market reaction.\nCatalyst: {res['name']} | type: {ev['catalyst_type']} | predicted {horizon}-day return: {O['pred_direction']} | actual: {ev['outcome']} → {verdict}\nAgent reasoning:\n{reasoning}\n\n{guide}\n\nA causal skill MUST be:\n  (a) STABLE — phrased to apply to other similar catalysts across companies/sectors;\n  (b) FALSIFIABLE — testable on future events (no vague "consider market sentiment");\n  (c) CAUSAL — link a feature (X) to expected return direction (Y), not surface correlation.\n\nReturn JSON ONLY:\n{{"skills":[{{"name":"<short, <10 words>",\n              "description":"<IF X THEN Y because Z — 1-2 sentences>",\n              "domain":"<one of: {_DISTILL_DOMAINS}>",\n              "stability_hint":"<one of: stable_across_sectors / sector_specific / regime_dependent / company_specific>",\n              "tags":["..."]}}]}}"""
    pkey = cache.key(prompt + '\x00' + cfg.PROMPT_VERSION, '', ['distill'])
    cached = cache.get(pkey)
    if cached is None:
        r = with_retry(lambda : _chat([{'role': 'user', 'content': prompt}], cfg.DISTILL_MAX_TOKENS), max_retry=cfg.LLM_MAX_RETRY, backoff=cfg.LLM_RETRY_BACKOFF, what='distill')
        cached = r.choices[0].message.content or ''
        cache.set(pkey, cached)
    import re
    m = re.search('\\{.*\\}', cached, re.S)
    try:
        parsed = json.loads(m.group(0) if m else cached)
        items = parsed.get('skills') or parsed.get('experiences', [])
        out = []
        for it in items:
            stability_hint = it.get('stability_hint', 'regime_dependent')
            if stability_hint not in ('stable_across_sectors', 'sector_specific', 'regime_dependent', 'company_specific'):
                stability_hint = 'regime_dependent'
            out.append({'name': (it.get('name') or '')[:80], 'description': (it.get('description') or '')[:500], 'domain': it.get('domain', 'rating'), 'stability_hint': stability_hint, 'tags': it.get('tags', [])})
        return out
    except Exception:
        return []
distill_experience = acquire_skill

def _classification_metrics(rows: list) -> dict:
    n = len(rows)
    if not n:
        return {'acc': 0.0, 'mcc': 0.0, 'f1_macro': 0.0}
    y = [1 if r['true_dir'] == 1 else 0 for r in rows]
    pred = [1 if r['conf'] >= 0.5 else 0 for r in rows]
    acc = sum((1 for (a, b) in zip(pred, y) if a == b)) / n
    tp = sum((1 for (a, b) in zip(pred, y) if a == 1 and b == 1))
    tn = sum((1 for (a, b) in zip(pred, y) if a == 0 and b == 0))
    fp = sum((1 for (a, b) in zip(pred, y) if a == 1 and b == 0))
    fn = sum((1 for (a, b) in zip(pred, y) if a == 0 and b == 1))
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / den if den else 0.0

    def _f1(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0
    f1_macro = (_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2
    return {'acc': round(acc, 4), 'mcc': round(mcc, 4), 'f1_macro': round(f1_macro, 4)}

def _bootstrap_ci(rows: list, n_boot: int=2000, seed: int=0) -> dict:
    import random
    n = len(rows)
    if n < 2:
        return {'acc_ci95': [None, None], 'mcc_ci95': [None, None], 'n_boot': 0}
    rng = random.Random(seed)
    (accs, mccs) = ([], [])
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        m = _classification_metrics(sample)
        accs.append(m['acc'])
        mccs.append(m['mcc'])

    def _pct(xs, q):
        xs = sorted(xs)
        k = (len(xs) - 1) * q
        lo = int(k)
        hi = min(lo + 1, len(xs) - 1)
        return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 4)
    return {'acc_ci95': [_pct(accs, 0.025), _pct(accs, 0.975)], 'mcc_ci95': [_pct(mccs, 0.025), _pct(mccs, 0.975)], 'n_boot': n_boot}
LLM_PRICES = {'gpt-4o-mini': (0.15, 0.6), 'gpt-4o': (2.5, 10.0), 'gpt-4.1-mini': (0.4, 1.6), 'gpt-4.1': (2.0, 8.0), 'o4-mini': (1.1, 4.4), 'deepseek-chat': (0.27, 1.1)}

def _token_cost(rows: list, model: str | None) -> dict:
    n_total = len(rows)
    measured = [r for r in rows if r.get('pt', 0) + r.get('ct', 0) > 0]
    nw = len(measured) or 1
    pt = sum((r.get('pt', 0) for r in measured))
    ct = sum((r.get('ct', 0) for r in measured))
    price = LLM_PRICES.get(model or cfg.LLM_MODEL)
    cost = round(pt / 1000000.0 * price[0] + ct / 1000000.0 * price[1], 4) if price else None
    return {'model': model or cfg.LLM_MODEL, 'n_measured': len(measured), 'n_total': n_total, 'complete': len(measured) == n_total, 'input_tokens': pt, 'output_tokens': ct, 'total_tokens': pt + ct, 'input_per_sample': round(pt / nw, 1), 'output_per_sample': round(ct / nw, 1), 'total_per_sample': round((pt + ct) / nw, 1), 'cost_usd': cost, 'cost_per_sample_usd': round(cost / nw, 6) if cost is not None else None}

def summarize(rows: list, experience_final_size: int | None=None, model: str | None=None) -> dict:
    n = len(rows)
    acc = sum((r['correct'] for r in rows)) / n if n else 0.0
    return {'n_catalysts': n, 'accuracy': round(acc, 3), **_classification_metrics(rows), **_bootstrap_ci(rows), 'token_cost': _token_cost(rows, model), 'experience_final_size': experience_final_size, 'built_at': time.strftime('%Y-%m-%d %H:%M')}

def print_summary(s: dict):
    print('\n' + '=' * 60)
    ci = s.get('acc_ci95', [None, None])
    ci_s = f' [95%CI {ci[0]:.3f}–{ci[1]:.3f}]' if ci and ci[0] is not None else ''
    print(f"acc={s['accuracy']:.1%}{ci_s}  MCC={s.get('mcc', 0):+.3f}  F1={s.get('f1_macro', 0):.3f}  ({s['n_catalysts']}events)")
    tc = s.get('token_cost', {})
    if tc:
        print(f"💰 {tc.get('model')}:  {tc.get('input_tokens')} /  {tc.get('output_tokens')} /  {tc.get('total_tokens')} tokens ( {tc.get('total_per_sample')}/)" + (f"  ≈ ${tc.get('cost_usd')}" if tc.get('cost_usd') is not None else ''))

def main():
    t0 = time.time()
    catalysts_path = _arg_val('--catalysts')
    DB = resolve_kb(_arg_val('--kb'))
    print(f' KB = databases/{DB.name}/')
    ss = SampleStore(str(DB / '01_info'))
    tk = TextKG(str(DB / '02_event'))
    EXP_DB = resolve_kb(_arg_val('--exp-kb')) if _arg_val('--exp-kb') else DB
    el = ExperienceLibrary(str(EXP_DB / '03_experience'))
    if EXP_DB != DB:
        print(f' ExpKB = databases/{EXP_DB.name}/ (events {DB.name} , )')
    if '--reset-exp' in __import__('sys').argv:
        el.exp = {}
        el.save()
        print('(--reset-exp)')
    cache = Cache(str(DB / '04_cache'))
    (catalysts, DATASET) = load_catalysts(catalysts_path, ss, DB)
    split = _arg_val('--split')
    build_exp = '--build-exp' in __import__('sys').argv
    if build_exp:
        assert EXP_DB == DB, '--build-exp ( --exp-kb )'
        assert catalysts_path is not None, '--build-exp  --catalysts <expr file>'
    readonly_exp = not build_exp and (catalysts_path is not None or split == 'test' or EXP_DB != DB)
    catalysts = sorted(catalysts, key=lambda e: e['decision_time'])
    retrieved = {}
    if readonly_exp:
        print(f': ,  {len(el)} (clear)')
        rp = retrieved_path(DB, DATASET)
        if rp.exists():
            retrieved = json.loads(rp.read_text(encoding='utf-8'))
            print(f'retrieve retrieved/{DATASET}.json: {len(retrieved)} events(eval retrieve)')
    print(f'inference {len(catalysts)} events…')
    rows = []

    def score_case(ev, res) -> dict:
        T = ev['decision_time']
        n_exp = sum((1 for e in el.exp.values() if e.get('resolve_date', '') < T))
        true_dir = direction(ev['outcome'])
        conf = res['OUTPUT']['pred_conf_verbal']
        correct = (conf >= 0.5) == (true_dir == 1)
        case = {'catalyst_id': ev['catalyst_id'], 'meta': {'stock_id': ev['stock_id'], 'stock_name': res['name'], 'catalyst_type': ev['catalyst_type'], 'decision_time': T, 'resolve_date': ev['resolve_date'], 'industry': res['industry'], 'n_experiences_available': n_exp}, 'INPUT': res['INPUT'], 'OUTPUT': res['OUTPUT'], 'GROUND_TRUTH': {'outcome': ev['outcome'], 'true_direction': 'positive' if true_dir == 1 else 'negative', 'fwd_ret': ev.get('fwd_ret')}, 'EVAL': {'correct': correct}, 'run_at': time.strftime('%Y-%m-%d %H:%M:%S')}
        save_case(case, DATASET, cfg.LLM_MODEL)
        u = res['OUTPUT'].get('usage', {})
        return {'catalyst_id': ev['catalyst_id'], 'etype': ev['catalyst_type'], 'correct': correct, 'conf': conf, 'true_dir': true_dir, 'n_exp': n_exp, 'pt': u.get('prompt_tokens', 0), 'ct': u.get('completion_tokens', 0), 'fwd_ret': ev.get('fwd_ret')}
    if readonly_exp:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        workers = min(len(catalysts), cfg.LLM_WORKERS) or 1
        results = {}
        with ThreadPoolExecutor(max_workers=workers) as exe:
            futs = {exe.submit(predict_catalyst, ev, ss, tk, el, cache, retrieved.get(ev['catalyst_id'])): ev['catalyst_id'] for ev in catalysts}
            for fut in tqdm(as_completed(futs), total=len(catalysts), desc=f'🔮 eval(concurrentx{workers})', unit='ev'):
                results[futs[fut]] = fut.result()
        for ev in catalysts:
            rows.append(score_case(ev, results[ev['catalyst_id']]))
    else:
        pbar = tqdm(catalysts, desc='🔮 train()', unit='ev')
        for ev in pbar:
            res = predict_catalyst(ev, ss, tk, el, cache)
            row = score_case(ev, res)
            rows.append(row)
            for L in distill_experience(ev, res, row['correct'], cache):
                if L.get('name') and L.get('description'):
                    ancestor = {
                        'source_catalyst_id': ev['catalyst_id'],
                        'decision_time': ev['decision_time'],
                        'resolve_time': ev['resolve_date'],
                        'outcome_source_hash': hashlib.sha256(json.dumps({
                            'catalyst_id': ev['catalyst_id'],
                            'outcome': ev['outcome'],
                            'fwd_ret': ev.get('fwd_ret')}, sort_keys=True,
                            ensure_ascii=False).encode('utf-8')).hexdigest(),
                        'distillation_prompt_hash': hashlib.sha256(json.dumps(
                            L, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest(),
                    }
                    el.evoke_skill(
                        L['name'], L['description'],
                        L.get('domain', ev['catalyst_type']),
                        advantage=0.2 if row['correct'] else -0.1,
                        resolve_date=ev['resolve_date'], tags=L.get('tags', []),
                        source=ev['catalyst_id'],
                        stability_hint=L.get('stability_hint', 'regime_dependent'),
                        provenance_ancestor=ancestor)
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix(skills=len(el), acc=f"{sum((r['correct'] for r in rows)) / len(rows):.0%}")
    summary = summarize(rows, experience_final_size=len(el), model=cfg.LLM_MODEL)
    cr = case_root(DATASET, cfg.LLM_MODEL)
    cr.mkdir(parents=True, exist_ok=True)
    (cr / '_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print_summary(summary)
    print(f"case log: {summary['n_catalysts']}  {cr.relative_to(BASE)}/  + _summary.json")
    print(f'({time.time() - t0:.0f}s)')
if __name__ == '__main__':
    main()
