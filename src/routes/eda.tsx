// src/routes/eda.tsx
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Fragment, useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Database, Columns3, AlertTriangle, Sparkles, ArrowRight } from "lucide-react";
import {
  getStore,
  colStats,
  histogram,
  corrMatrix,
  missingCount,
  type CsvStore,
} from "@/lib/csv-store";

export const Route = createFileRoute("/eda")({
  head: () => ({
    meta: [
      { title: "EDA — Lumen AI Data Studio" },
      {
        name: "description",
        content: "Exploratory data analysis: stats, correlations, distributions.",
      },
    ],
  }),
  component: EDA,
});

function corrColor(v: number) {
  const a = Math.abs(v);
  if (v >= 0) return `oklch(0.55 0.22 264 / ${0.15 + a * 0.75})`;
  return `oklch(0.62 0.22 25 / ${0.15 + a * 0.75})`;
}

function qualityScore(missingPct: number): string {
  if (missingPct < 1) return "A+";
  if (missingPct < 3) return "A−";
  if (missingPct < 8) return "B";
  if (missingPct < 15) return "C";
  return "D";
}

function EDA() {
  const navigate = useNavigate();
  const [store, setStore] = useState<CsvStore | null>(null);

  useEffect(() => {
    setStore(getStore());
  }, []);

  // Veri yoksa fallback
  if (!store) {
    return (
      <div className="mx-auto max-w-6xl space-y-6 p-6">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Exploratory analysis</h1>
          <p className="text-sm text-muted-foreground">
            Henüz veri yüklenmedi.
          </p>
        </div>
        <Button onClick={() => navigate({ to: "/" })}>← Upload a dataset</Button>
      </div>
    );
  }

  const { headers, rows, numericCols, categoricalCols, fileName } = store;

  // İstatistikler
  const missing = missingCount(rows, headers);
  const totalCells = rows.length * headers.length;
  const totalMissing = Object.values(missing).reduce((a, b) => a + b, 0);
  const missingPct = totalCells > 0 ? (totalMissing / totalCells) * 100 : 0;

  const STATS = [
    {
      label: "Rows",
      value: rows.length.toLocaleString(),
      icon: Database,
      hint: `${headers.length} columns total`,
    },
    {
      label: "Columns",
      value: String(headers.length),
      icon: Columns3,
      hint: `${categoricalCols.length} categorical · ${numericCols.length} numeric`,
    },
    {
      label: "Missing values",
      value: `${missingPct.toFixed(1)}%`,
      icon: AlertTriangle,
      hint: `${totalMissing} cells`,
    },
    {
      label: "Quality score",
      value: qualityScore(missingPct),
      icon: Sparkles,
      hint: missingPct < 5 ? "ready for modeling" : "consider imputation",
    },
  ];

  // Korelasyon — max 7 kolon göster (performans)
  const corrCols = numericCols.slice(0, 7);
  const matrix = corrCols.length > 1 ? corrMatrix(rows, corrCols, headers) : [];

  // Dağılımlar — ilk 4 sayısal kolon
  const distCols = numericCols.slice(0, 4);

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Exploratory analysis</h1>
          <p className="text-sm text-muted-foreground">
            {fileName} · {rows.length.toLocaleString()} rows profiled
          </p>
        </div>
        <Button className="gap-2" onClick={() => navigate({ to: "/predict" })}>
          Predict <ArrowRight className="h-4 w-4" />
        </Button>
      </div>

      {/* Stat cards */}
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

      {/* Kolon özeti */}
      <Card>
        <CardHeader>
          <CardTitle>Column summary</CardTitle>
          <CardDescription>Min / Mean / Max for numeric columns</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  {["Column", "Type", "Min", "Mean", "Max", "Missing"].map((h) => (
                    <th key={h} className="pb-2 pr-4 text-left font-mono uppercase tracking-wider text-muted-foreground">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {headers.map((col) => {
                  const idx = headers.indexOf(col);
                  const isNum = numericCols.includes(col);
                  const stats = isNum ? colStats(rows, idx) : null;
                  return (
                    <tr key={col} className="border-b border-border/50 last:border-0">
                      <td className="py-2 pr-4 font-mono font-medium">{col}</td>
                      <td className="py-2 pr-4">
                        <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${isNum ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"}`}>
                          {isNum ? "numeric" : "categorical"}
                        </span>
                      </td>
                      <td className="py-2 pr-4 font-mono">{stats ? stats.min.toLocaleString() : "—"}</td>
                      <td className="py-2 pr-4 font-mono">{stats ? stats.mean.toFixed(1) : "—"}</td>
                      <td className="py-2 pr-4 font-mono">{stats ? stats.max.toLocaleString() : "—"}</td>
                      <td className="py-2 pr-4 font-mono">{missing[col] ?? 0}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Korelasyon heatmap */}
      {matrix.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Correlation heatmap</CardTitle>
            <CardDescription>
              Pearson correlation · {corrCols.length} numeric columns
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <div
                className="grid gap-1"
                style={{
                  gridTemplateColumns: `120px repeat(${corrCols.length}, minmax(48px, 1fr))`,
                }}
              >
                <div />
                {corrCols.map((c) => (
                  <div
                    key={c}
                    className="px-1 pb-2 text-center font-mono text-[11px] text-muted-foreground"
                  >
                    {c}
                  </div>
                ))}
                {matrix.map((row, i) => (
                  <Fragment key={`row-${i}`}>
                    <div className="flex items-center justify-end pr-3 font-mono text-[11px] text-muted-foreground">
                      {corrCols[i]}
                    </div>
                    {row.map((v, j) => (
                      <div
                        key={`${i}-${j}`}
                        className="aspect-square flex items-center justify-center rounded-sm text-[10px] font-mono text-foreground/90"
                        style={{ backgroundColor: corrColor(v) }}
                        title={`${corrCols[i]} × ${corrCols[j]}: ${v}`}
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
      )}

      {/* Dağılım grafikleri */}
      {distCols.length > 0 && (
        <div className="grid gap-4 md:grid-cols-2">
          {distCols.map((col) => {
            const idx = headers.indexOf(col);
            const bars = histogram(rows, idx);
            const stats = colStats(rows, idx);
            const max = Math.max(...bars);
            return (
              <Card key={col}>
                <CardHeader>
                  <CardTitle className="text-base font-mono">{col}</CardTitle>
                  <CardDescription>
                    Distribution · 8 bins
                    {stats ? ` · mean ${stats.mean.toFixed(1)}` : ""}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex h-36 items-end gap-2">
                    {bars.map((b, i) => (
                      <div
                        key={i}
                        className="flex-1 rounded-t-sm transition-all"
                        style={{
                          height: `${max > 0 ? (b / max) * 100 : 0}%`,
                          backgroundImage: "var(--gradient-primary)",
                        }}
                      />
                    ))}
                  </div>
                  <div className="mt-2 flex justify-between font-mono text-[10px] text-muted-foreground">
                    <span>{stats?.min.toLocaleString()}</span>
                    <span>{stats?.median.toLocaleString()}</span>
                    <span>{stats?.max.toLocaleString()}</span>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}