'use client';

import { useMemo, useState } from 'react';

type ModelName = 'camembert' | 'gpt2';
type ExplainMethod = 'none' | 'lime' | 'shap' | 'ig';

type PredictResponse = {
  model: ModelName;
  probs: { label: 'non-toxic' | 'toxic'; score: number }[];
  toxic_score: number;
};

type ExplainResponse =
  | {
      model: ModelName;
      method: 'lime' | 'shap';
      toxic_score: number;
      html: string;
      attributions?: never;
    }
  | {
      model: ModelName;
      method: 'ig';
      toxic_score: number;
      attributions: { token: string; score: number }[];
      html?: never;
    };

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? 'http://127.0.0.1:8000';
const fmtPct = (x: number) => `${(x * 100).toFixed(1)}%`;

/** Iframe pour rendre le HTML LIME/SHAP (scripts inclus) */
function HtmlSandbox({ html, height = 520 }: { html: string; height?: number }) {
  return (
    <iframe
      title="explanation"
      className="w-full rounded-xl border border-white/20 bg-white/5 shadow-inner"
      style={{ height }}
      sandbox="allow-scripts"
      srcDoc={html}
    />
  );
}

export default function Page() {
  const [text, setText] = useState('Tu es bête et idiot.');
  const [model, setModel] = useState<ModelName>('camembert');
  const [method, setMethod] = useState<ExplainMethod>('none');
  const [loading, setLoading] = useState(false);

  const [predict, setPredict] = useState<PredictResponse | null>(null);
  const [explain, setExplain] = useState<ExplainResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const toxicProb = predict?.probs.find((p) => p.label === 'toxic')?.score ?? 0;
  const label = toxicProb > 0.5 ? 'Toxique' : 'Non toxique';
  const canExplain = method !== 'none' && !!predict;

  async function handleRun() {
    setLoading(true);
    setPredict(null);
    setExplain(null);
    setError(null);

    try {
      const pRes = await fetch(`${API_BASE}/v1/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model }),
      });
      if (!pRes.ok) throw new Error(`Predict failed: ${pRes.status}`);
      const pred: PredictResponse = await pRes.json();
      setPredict(pred);

      if (method !== 'none') {
        const payloadMethod: 'lime' | 'shap' | 'ig' = method === 'ig' ? 'ig' : method;
        const eRes = await fetch(`${API_BASE}/v1/explain`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, model, method: payloadMethod }),
        });
        if (!eRes.ok) throw new Error(`Explain failed: ${eRes.status}`);
        const data: ExplainResponse = await eRes.json();
        setExplain(data);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  const igSorted = useMemo(() => {
    if (!explain || explain.method !== 'ig') return [];
    return [...explain.attributions].sort((a, b) => b.score - a.score).slice(0, 30);
  }, [explain]);

  const modelId = 'model-select';
  const methodId = 'method-select';
  const textareaId = 'text-input';

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* BACKGROUND — gradient animé + mesh */}
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute -top-32 -left-16 h-[36rem] w-[36rem] rounded-full blur-3xl opacity-30 bg-[radial-gradient(circle_at_30%_30%,#f43f5e,transparent_60%)]" />
        <div className="absolute top-10 right-0 h-[36rem] w-[36rem] rounded-full blur-3xl opacity-30 bg-[radial-gradient(circle_at_70%_30%,#8b5cf6,transparent_60%)]" />
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 h-[36rem] w-[36rem] rounded-full blur-3xl opacity-30 bg-[radial-gradient(circle_at_50%_80%,#06b6d4,transparent_60%)]" />
        <div className="absolute inset-0 bg-[linear-gradient(120deg,rgba(255,255,255,.9),rgba(255,255,255,.6))]" />
      </div>

      <div className="mx-auto max-w-6xl px-4 py-10">
        {/* HEADER — signature */}
        <header className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">
              <span className="bg-clip-text text-transparent bg-[linear-gradient(90deg,#f43f5e,#8b5cf6,#06b6d4)]">
                Toxicité
              </span>{' '}
              <span className="text-neutral-600">— French Démo</span>
            </h1>
            <p className="mt-1 text-sm text-neutral-600">
              CamemBERT / GPT-2 · LIME / SHAP / Integrated Gradients · API FastAPI
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="rounded-full bg-white/70 backdrop-blur-md px-3 py-1 text-xs font-medium shadow-sm border border-white/50">
              Designed by <span className="font-semibold">Anass Ait Lasri</span>
            </div>
            <div className="h-9 w-9 rounded-full bg-[conic-gradient(from_180deg_at_50%_50%,#f43f5e,#8b5cf6,#06b6d4,#f43f5e)] p-[2px] shadow">
              <div className="flex h-full w-full items-center justify-center rounded-full bg-white/90 text-[11px] font-bold text-neutral-800">
                AAL
              </div>
            </div>
          </div>
        </header>

        {/* INPUT — carte glass */}
        <section className="rounded-3xl border border-white/40 bg-white/70 backdrop-blur-xl shadow-xl p-6">
          <div className="flex flex-col gap-4">
            <label htmlFor={textareaId} className="text-sm font-medium text-neutral-800">
              Texte à analyser
            </label>
            <textarea
              id={textareaId}
              className="w-full resize-y rounded-2xl border border-neutral-200/70 bg-white/90 px-4 py-3 text-sm outline-none transition focus:ring-2 focus:ring-fuchsia-300"
              rows={5}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Tape ici ta phrase…"
            />

            <div className="grid gap-4 md:grid-cols-3">
              <div className="flex flex-col gap-2">
                <label htmlFor={modelId} className="text-sm font-medium text-neutral-800">
                  Modèle
                </label>
                <select
                  id={modelId}
                  className="rounded-2xl border border-neutral-200/70 bg-white/90 px-4 py-2.5 text-sm"
                  value={model}
                  onChange={(e) => setModel(e.target.value as ModelName)}
                >
                  <option value="camembert">CamemBERT</option>
                  <option value="gpt2">GPT-2</option>
                </select>
              </div>

              <div className="flex flex-col gap-2">
                <label htmlFor={methodId} className="text-sm font-medium text-neutral-800">
                  Explication
                </label>
                <select
                  id={methodId}
                  className="rounded-2xl border border-neutral-200/70 bg-white/90 px-4 py-2.5 text-sm"
                  value={method}
                  onChange={(e) => setMethod(e.target.value as ExplainMethod)}
                >
                  <option value="none">Aucune</option>
                  <option value="lime">LIME</option>
                  <option value="shap">SHAP</option>
                  <option value="ig">Integrated Gradients</option>
                </select>
              </div>

              <div className="flex items-end">
                <button
                  onClick={handleRun}
                  disabled={loading || !text.trim()}
                  className="group inline-flex w-full items-center justify-center rounded-2xl px-5 py-3 text-sm font-medium text-white shadow-lg transition
                             bg-[linear-gradient(90deg,#f43f5e,#8b5cf6,#06b6d4)]
                             hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? (
                    <span className="flex items-center gap-2">
                      <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/50 border-t-white" />
                      Analyse…
                    </span>
                  ) : (
                    'Analyser'
                  )}
                </button>
              </div>
            </div>

            {error && (
              <div
                role="alert"
                aria-live="polite"
                className="rounded-2xl border border-rose-200 bg-rose-50/80 px-4 py-2 text-sm text-rose-800"
              >
                {error}
              </div>
            )}
          </div>
        </section>

        {/* RÉSULTATS */}
        {predict && (
          <section className="mt-6 grid gap-6 md:grid-cols-2">
            {/* Score */}
            <div className="rounded-3xl border border-white/40 bg-white/70 backdrop-blur-xl shadow-xl p-6">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-sm font-medium text-neutral-800">Résultat</h2>
                <span
                  className={`rounded-full px-3 py-1 text-xs font-semibold shadow-sm border
                    ${
                      toxicProb > 0.5
                        ? 'bg-rose-100/80 text-rose-700 border-rose-200'
                        : 'bg-emerald-100/80 text-emerald-700 border-emerald-200'
                    }`}
                >
                  {label}
                </span>
              </div>

              <div className="flex items-center justify-between text-sm text-neutral-700">
                <div>
                  Toxicité&nbsp;<span className="font-semibold">{fmtPct(predict.toxic_score)}</span>
                </div>
                <div className="text-xs text-neutral-500">Modèle&nbsp;: {predict.model}</div>
              </div>

              <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-neutral-200/70">
                <div
                  className="h-full transition-all duration-500"
                  style={{
                    width: `${Math.min(100, Math.max(0, toxicProb * 100))}%`,
                    background:
                      toxicProb > 0.5
                        ? 'linear-gradient(90deg,#f43f5e,#fb7185)'
                        : 'linear-gradient(90deg,#10b981,#34d399)',
                  }}
                />
              </div>

              <div className="mt-4 grid grid-cols-2 gap-2 text-sm">
                {predict.probs.map((p) => (
                  <div
                    key={p.label}
                    className="flex items-center justify-between rounded-xl border border-neutral-200/70 bg-white/90 px-3 py-2"
                  >
                    <span className="capitalize">{p.label.replace('-', ' ')}</span>
                    <span className="font-semibold">{fmtPct(p.score)}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Explication */}
            <div className="rounded-3xl border border-white/40 bg-white/70 backdrop-blur-xl shadow-xl p-6">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-sm font-medium text-neutral-800">Explication</h2>
                <span className="text-xs text-neutral-500">
                  {method === 'none' ? '—' : method.toUpperCase()}
                </span>
              </div>

              {!canExplain && (
                <p className="text-sm text-neutral-600">
                  Choisis une méthode d’explication puis relance l’analyse.
                </p>
              )}

              {explain && (explain.method === 'lime' || explain.method === 'shap') && explain.html && (
                <HtmlSandbox html={explain.html} />
              )}

              {explain && explain.method === 'ig' && (
                <div className="space-y-3">
                  <p className="text-sm text-neutral-600">Top 30 tokens contributifs (scores normalisés).</p>
                  <div className="flex flex-wrap gap-2">
                    {igSorted.map((t, i) => (
                      <span
                        key={`${t.token}-${i}`}
                        className="rounded-full border border-white/30 bg-white/70 backdrop-blur px-3 py-1 text-xs shadow-sm"
                        title={t.score.toFixed(4)}
                        style={{
                          background:
                            t.score > 0.66
                              ? 'linear-gradient(90deg,#fee2e2,#ffe4e6)'
                              : t.score > 0.33
                              ? 'linear-gradient(90deg,#fef3c7,#ffedd5)'
                              : 'linear-gradient(90deg,#dcfce7,#d1fae5)',
                        }}
                      >
                        {t.token} · {t.score.toFixed(2)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {/* FOOTER — signature */}
        <footer className="mt-10 text-center text-xs text-neutral-600">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/50 bg-white/70 px-3 py-1 backdrop-blur-md shadow-sm">
            <span>© {new Date().getFullYear()}</span>
            <span className="font-semibold">Anass Ait Lasri</span>
            <span className="h-1 w-1 rounded-full bg-neutral-400" />
            <span>
              API&nbsp;
              <code className="rounded bg-neutral-100/80 px-1">{API_BASE}</code>
            </span>
          </div>
        </footer>
      </div>
    </main>
  );
}
