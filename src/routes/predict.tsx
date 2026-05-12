import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Sparkles, TrendingUp, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export const Route = createFileRoute("/predict")({
  head: () => ({
    meta: [
      { title: "Predict — Lumen AI Data Studio" },
      { name: "description", content: "Run model predictions with dynamic feature inputs." },
    ],
  }),
  component: Predict,
});

type Field =
  | { key: string; label: string; type: "number"; placeholder: string }
  | { key: string; label: string; type: "select"; options: string[] };

const FIELDS: Field[] = [
  { key: "age", label: "Age", type: "number", placeholder: "34" },
  { key: "income", label: "Annual income", type: "number", placeholder: "72400" },
  { key: "score", label: "Engagement score", type: "number", placeholder: "812" },
  { key: "tenure", label: "Tenure (months)", type: "number", placeholder: "18" },
  { key: "segment", label: "Segment", type: "select", options: ["trial", "growth", "core", "enterprise"] },
  { key: "channel", label: "Acquisition channel", type: "select", options: ["organic", "paid", "referral", "partner"] },
];

function Predict() {
  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<{ label: string; confidence: number } | null>(null);
  const [loading, setLoading] = useState(false);

  const onPredict = () => {
    setLoading(true);
    setTimeout(() => {
      setResult({ label: "Will retain", confidence: 0.92 });
      setLoading(false);
    }, 700);
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Run a prediction</h1>
        <p className="text-sm text-muted-foreground">
          Model · churn-classifier-v3 · trained on customers_sample.csv
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
        <Card>
          <CardHeader>
            <CardTitle>Input features</CardTitle>
            <CardDescription>Fields are inferred from your dataset schema.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 sm:grid-cols-2">
              {FIELDS.map((f) => (
                <div key={f.key} className="space-y-2">
                  <Label htmlFor={f.key} className="text-xs uppercase tracking-wider text-muted-foreground">
                    {f.label}
                  </Label>
                  {f.type === "number" ? (
                    <Input
                      id={f.key}
                      type="number"
                      placeholder={f.placeholder}
                      value={values[f.key] ?? ""}
                      onChange={(e) => setValues((v) => ({ ...v, [f.key]: e.target.value }))}
                    />
                  ) : (
                    <Select
                      value={values[f.key]}
                      onValueChange={(val) => setValues((v) => ({ ...v, [f.key]: val }))}
                    >
                      <SelectTrigger id={f.key}>
                        <SelectValue placeholder="Select…" />
                      </SelectTrigger>
                      <SelectContent>
                        {f.options.map((o) => (
                          <SelectItem key={o} value={o}>
                            {o}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                </div>
              ))}
            </div>
            <div className="mt-6 flex items-center justify-end gap-3">
              <Button variant="outline" onClick={() => { setValues({}); setResult(null); }}>
                Reset
              </Button>
              <Button onClick={onPredict} disabled={loading} className="gap-2">
                <Sparkles className="h-4 w-4" />
                {loading ? "Predicting…" : "Predict"}
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card
          className="overflow-hidden"
          style={{ backgroundImage: result ? "var(--gradient-surface)" : undefined }}
        >
          <CardHeader>
            <CardTitle>Result</CardTitle>
            <CardDescription>
              {result ? "Latest inference" : "Submit features to see prediction"}
            </CardDescription>
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
                    <p className="text-xl font-semibold">{result.label}</p>
                  </div>
                </div>

                <div>
                  <div className="mb-2 flex items-center justify-between text-xs">
                    <span className="uppercase tracking-wider text-muted-foreground">Confidence</span>
                    <span className="font-mono">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-muted">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${result.confidence * 100}%`,
                        backgroundImage: "var(--gradient-primary)",
                      }}
                    />
                  </div>
                </div>

                <div className="rounded-md border border-border bg-card/50 p-3 text-xs text-muted-foreground">
                  <div className="mb-1 flex items-center gap-1.5 text-foreground">
                    <TrendingUp className="h-3.5 w-3.5 text-primary" />
                    Top contributing features
                  </div>
                  <ul className="space-y-1 font-mono">
                    <li className="flex justify-between"><span>tenure</span><span>+0.34</span></li>
                    <li className="flex justify-between"><span>score</span><span>+0.21</span></li>
                    <li className="flex justify-between"><span>income</span><span>+0.12</span></li>
                  </ul>
                </div>
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
    </div>
  );
}