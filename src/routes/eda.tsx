import { createFileRoute } from "@tanstack/react-router";
import { Fragment } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Database, Columns3, AlertTriangle, Sparkles } from "lucide-react";

export const Route = createFileRoute("/eda")({
  head: () => ({
    meta: [
      { title: "EDA — Lumen AI Data Studio" },
      { name: "description", content: "Exploratory data analysis: stats, correlations, distributions." },
    ],
  }),
  component: EDA,
});

const STATS = [
  { label: "Rows", value: "12,480", icon: Database, hint: "+1,204 vs last upload" },
  { label: "Columns", value: "18", icon: Columns3, hint: "4 categorical · 14 numeric" },
  { label: "Missing values", value: "2.1%", icon: AlertTriangle, hint: "264 cells" },
  { label: "Quality score", value: "A−", icon: Sparkles, hint: "ready for modeling" },
];

const COLS = ["age", "income", "score", "tenure", "visits", "spend", "churn"];

const DISTRIBUTIONS = [
  { name: "age", bars: [4, 9, 18, 28, 22, 14, 8, 4] },
  { name: "income", bars: [3, 7, 14, 22, 26, 18, 9, 4] },
  { name: "score", bars: [2, 5, 11, 19, 24, 21, 12, 6] },
  { name: "tenure", bars: [12, 18, 22, 17, 13, 9, 6, 3] },
];

function corrColor(v: number) {
  // v in [-1, 1] -> blue intensity
  const a = Math.abs(v);
  if (v >= 0) return `oklch(0.55 0.22 264 / ${0.15 + a * 0.75})`;
  return `oklch(0.62 0.22 25 / ${0.15 + a * 0.75})`;
}

function EDA() {
  // deterministic pseudo-correlation matrix
  const matrix = COLS.map((_, i) =>
    COLS.map((_, j) => {
      if (i === j) return 1;
      const v = Math.sin((i + 1) * (j + 1) * 1.7) * 0.85;
      return Math.round(v * 100) / 100;
    }),
  );

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Exploratory analysis</h1>
        <p className="text-sm text-muted-foreground">
          customers_sample.csv · profiled in 1.2s
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {STATS.map((s) => (
          <Card key={s.label}>
            <CardContent className="p-5">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-xs uppercase tracking-wider text-muted-foreground">
                    {s.label}
                  </p>
                  <p className="mt-1 text-2xl font-semibold tracking-tight">{s.value}</p>
                  <p className="mt-2 text-xs text-muted-foreground">{s.hint}</p>
                </div>
                <div className="flex h-9 w-9 items-center justify-center rounded-md bg-primary/10 text-primary">
                  <s.icon className="h-4 w-4" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Correlation heatmap</CardTitle>
          <CardDescription>Pearson correlation across numeric features</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <div
              className="grid gap-1"
              style={{ gridTemplateColumns: `120px repeat(${COLS.length}, minmax(48px, 1fr))` }}
            >
              <div />
              {COLS.map((c) => (
                <div key={c} className="px-1 pb-2 text-center font-mono text-[11px] text-muted-foreground">
                  {c}
                </div>
              ))}
              {matrix.map((row, i) => (
                <Fragment key={`row-${i}`}>
                  <div className="flex items-center justify-end pr-3 font-mono text-[11px] text-muted-foreground">
                    {COLS[i]}
                  </div>
                  {row.map((v, j) => (
                    <div
                      key={`${i}-${j}`}
                      className="aspect-square rounded-sm flex items-center justify-center text-[10px] font-mono text-foreground/90"
                      style={{ backgroundColor: corrColor(v) }}
                      title={`${COLS[i]} × ${COLS[j]}: ${v}`}
                    >
                      {v.toFixed(2)}
                    </div>
                  ))}
                </Fragment>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        {DISTRIBUTIONS.map((d) => (
          <Card key={d.name}>
            <CardHeader>
              <CardTitle className="text-base font-mono">{d.name}</CardTitle>
              <CardDescription>Distribution · 8 bins</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex h-36 items-end gap-2">
                {d.bars.map((b, i) => (
                  <div key={i} className="flex-1 rounded-t-sm" style={{ height: `${b * 3}px`, backgroundImage: "var(--gradient-primary)" }} />
                ))}
              </div>
              <div className="mt-2 flex justify-between font-mono text-[10px] text-muted-foreground">
                <span>min</span>
                <span>median</span>
                <span>max</span>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}