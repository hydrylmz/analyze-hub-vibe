// src/routes/predict.tsx
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import {
  Sparkles,
  TrendingUp,
  CheckCircle2,
  AlertCircle,
  Cpu,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getStore, type CsvStore } from "@/lib/csv-store";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export const Route = createFileRoute("/predict")({
  head: () => ({
    meta: [
      { title: "Predict — Lumen AI Data Studio" },
      { name: "description", content: "Run model predictions with dynamic feature inputs." },
    ],
  }),
  component: Predict,
});

type TrainResult = {
  task: string;
  target: string;
  metrics: Record<string, unknown>;
  feature_importance: Record<string, number>;
  train_rows: number;
  test_rows: number;
};

type PredictResult = {
  prediction: string | number;
  confidence: number | null;
  feature_importance: Record<string, number>;
};

function Predict() {
  const navigate = useNavigate();
  const [store, setStore] = useState<CsvStore | null>(null);

  // Train state
  const [target, setTarget] = useState<string>("");
  const [excluded, setExcluded] = useState<string[]>([]);
  const [excludeCol, setExcludeCol] = useState<string>("");
  const [training, setTraining] = useState(false);
  const [trainResult, setTrainResult] = useState<TrainResult | null>(null);

  // Predict state
  const [values, setValues] = useState<Record<string, string>>({});
  const [predicting, setPredicting] = useState(false);
  const [result, setResult] = useState<PredictResult | null>(null);

  const [error, setError] = useState<string | null>(null);

  useEffect(() => { setStore(getStore()); }, []);

  const addExclude = (col: string) => {
    if (!col || excluded.includes(col) || col === target) return;
    setExcluded((e) => [...e, col]);
    setExcludeCol("");
  };

  const removeExclude = (col: string) => {
    setExcluded((e) => e.filter((c) => c !== col));
  };

  const onTrain = async () => {
    if (!target) return;
    setTraining(true); setError(null); setTrainResult(null); setResult(null);
    try {
      const res = await fetch(`${API_URL}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target, exclude: excluded }),
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail ?? "Train hatası"); }
      setTrainResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Backend bağlantısı yok.");
    } finally { setTraining(false); }
  };

  const onPredict = async () => {
    setPredicting(true); setError(null); setResult(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail ?? "Predict hatası"); }
      setResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Backend bağlantısı yok.");
    } finally { setPredicting(false); }
  };

  if (!store) {
    return (
      <div className="mx-auto max-w-6xl space-y-6 p-6">
        <h1 className="text-2xl font-semibold tracking-tight">Run a prediction</h1>
        <p className="text-sm text-muted-foreground">Henüz veri yüklenmedi.</p>
        <Button onClick={() => navigate({ to: "/" })}>← Upload a dataset</Button>
      </div>
    );
  }

  const { headers, numericCols, categoricalCols, rows, fileName } = store;

  // Hedef ve exclude edilenler dışındaki kolonlar input olarak gösterilir
  const inputHeaders = trainResult
    ? headers.filter((h) => h !== trainResult.target && !excluded.includes(h))
    : headers;

  const categoricalOptions: Record<string, string[]> = {};
  categoricalCols.forEach((col) => {
    const idx = headers.indexOf(col);
    categoricalOptions[col] = [...new Set(rows.map((r) => r[idx]).filter(Boolean))].slice(0, 20);
  });

  // Target ve zaten excluded olanlar hariç exclude edilebilir kolonlar
  const excludableHeaders = headers.filter(
    (h) => h !== target && !excluded.includes(h)
  );

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Run a prediction</h1>
        <p className="text-sm text-muted-foreground">Dataset · {fileName} · {headers.length} features</p>
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          <AlertCircle className="h-4 w-4 shrink-0" />{error}
        </div>
      )}

      {/* ADIM 1: Train */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-primary" />Step 1 — Train model
          </CardTitle>
          <CardDescription>
            Hedef kolonu seç, gereksiz kolonları çıkar, modeli eğit.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Target + Train butonu */}
          <div className="flex flex-wrap items-end gap-4">
            <div className="flex-1 space-y-2 min-w-[180px]">
              <Label className="text-xs uppercase tracking-wider text-muted-foreground">
                Target column
              </Label>
              <Select value={target} onValueChange={(v) => {
                setTarget(v);
                setExcluded((e) => e.filter((c) => c !== v));
                setTrainResult(null);
                setResult(null);
              }}>
                <SelectTrigger><SelectValue placeholder="Select target…" /></SelectTrigger>
                <SelectContent>
                  {headers.map((h) => <SelectItem key={h} value={h}>{h}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={onTrain} disabled={!target || training} className="gap-2">
              <Cpu className="h-4 w-4" />{training ? "Training…" : "Train"}
            </Button>
          </div>

          {/* Exclude kolonlar */}
          {target && (
            <div className="space-y-2">
              <Label className="text-xs uppercase tracking-wider text-muted-foreground">
                Exclude columns (leakage, ID, gereksiz)
              </Label>
              <div className="flex flex-wrap gap-2">
                {excluded.map((col) => (
                  <Badge
                    key={col}
                    variant="secondary"
                    className="gap-1 cursor-pointer pr-1"
                    onClick={() => removeExclude(col)}
                  >
                    {col}
                    <X className="h-3 w-3" />
                  </Badge>
                ))}
                <div className="flex gap-2">
                  <Select value={excludeCol} onValueChange={setExcludeCol}>
                    <SelectTrigger className="h-7 w-48 text-xs">
                      <SelectValue placeholder="Add column…" />
                    </SelectTrigger>
                    <SelectContent>
                      {excludableHeaders.map((h) => (
                        <SelectItem key={h} value={h} className="text-xs">{h}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-7 text-xs"
                    onClick={() => addExclude(excludeCol)}
                    disabled={!excludeCol}
                  >
                    + Add
                  </Button>
                </div>
              </div>
            </div>
          )}

          {/* Train sonucu */}
          {trainResult && (
            <div className="rounded-md border border-border bg-muted/30 p-4 text-sm space-y-2">
              <p className="font-medium text-foreground">
                ✅ Model eğitildi — <span className="font-mono text-primary">{trainResult.task}</span>
              </p>
              <div className="flex flex-wrap gap-4 text-xs text-muted-foreground font-mono">
                {Object.entries(trainResult.metrics).map(([k, v]) => (
                  <span key={k}>{k}: <strong className="text-foreground">{String(v)}</strong></span>
                ))}
                <span>train rows: <strong className="text-foreground">{trainResult.train_rows}</strong></span>
                <span>test rows: <strong className="text-foreground">{trainResult.test_rows}</strong></span>
              </div>
              {Object.keys(trainResult.feature_importance).length > 0 && (
                <ul className="space-y-0.5 font-mono text-xs">
                  {Object.entries(trainResult.feature_importance).slice(0, 5).map(([k, v]) => (
                    <li key={k} className="flex justify-between max-w-xs">
                      <span>{k}</span><span className="text-primary">{v}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* ADIM 2: Predict */}
      {trainResult && (
        <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
          <Card>
            <CardHeader>
              <CardTitle>Step 2 — Input features</CardTitle>
              <CardDescription>
                Hedef: <span className="font-mono text-primary">{trainResult.target}</span>
                {excluded.length > 0 && (
                  <span className="ml-2 text-muted-foreground">
                    · {excluded.length} kolon exclude edildi
                  </span>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 sm:grid-cols-2">
                {inputHeaders.map((col) => {
                  const isNum = numericCols.includes(col);
                  const options = categoricalOptions[col] ?? [];
                  return (
                    <div key={col} className="space-y-2">
                      <Label htmlFor={col} className="text-xs uppercase tracking-wider text-muted-foreground">{col}</Label>
                      {isNum ? (
                        <Input id={col} type="number" placeholder="Enter value"
                          value={values[col] ?? ""}
                          onChange={(e) => setValues((v) => ({ ...v, [col]: e.target.value }))} />
                      ) : (
                        <Select value={values[col]} onValueChange={(val) => setValues((v) => ({ ...v, [col]: val }))}>
                          <SelectTrigger id={col}><SelectValue placeholder="Select…" /></SelectTrigger>
                          <SelectContent>
                            {options.map((o) => <SelectItem key={o} value={o}>{o}</SelectItem>)}
                          </SelectContent>
                        </Select>
                      )}
                    </div>
                  );
                })}
              </div>
              <div className="mt-6 flex items-center justify-end gap-3">
                <Button variant="outline" onClick={() => { setValues({}); setResult(null); }}>Reset</Button>
                <Button onClick={onPredict} disabled={predicting} className="gap-2">
                  <Sparkles className="h-4 w-4" />{predicting ? "Predicting…" : "Predict"}
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="overflow-hidden" style={{ backgroundImage: result ? "var(--gradient-surface)" : undefined }}>
            <CardHeader>
              <CardTitle>Result</CardTitle>
              <CardDescription>{result ? "Latest inference" : "Submit features to see prediction"}</CardDescription>
            </CardHeader>
            <CardContent>
              {result ? (
                <div className="space-y-5">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-md bg-success/15 text-success">
                      <CheckCircle2 className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-wider text-muted-foreground">Prediction</p>
                      <p className="text-xl font-semibold">{String(result.prediction)}</p>
                    </div>
                  </div>
                  {result.confidence !== null && (
                    <div>
                      <div className="mb-2 flex items-center justify-between text-xs">
                        <span className="uppercase tracking-wider text-muted-foreground">Confidence</span>
                        <span className="font-mono">{((result.confidence ?? 0) * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-muted">
                        <div className="h-full rounded-full" style={{ width: `${(result.confidence ?? 0) * 100}%`, backgroundImage: "var(--gradient-primary)" }} />
                      </div>
                    </div>
                  )}
                  {Object.keys(result.feature_importance).length > 0 && (
                    <div className="rounded-md border border-border bg-card/50 p-3 text-xs text-muted-foreground">
                      <div className="mb-1 flex items-center gap-1.5 text-foreground">
                        <TrendingUp className="h-3.5 w-3.5 text-primary" />Top contributing features
                      </div>
                      <ul className="space-y-1 font-mono">
                        {Object.entries(result.feature_importance).map(([k, v]) => (
                          <li key={k} className="flex justify-between">
                            <span>{k}</span><span>{v > 0 ? "+" : ""}{v}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex h-48 flex-col items-center justify-center text-center text-sm text-muted-foreground">
                  <Sparkles className="mb-2 h-6 w-6 text-primary/60" />
                  Fill in features and run the model.
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}