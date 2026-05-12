import { createFileRoute } from "@tanstack/react-router";
import { Download, TrendingUp, TrendingDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export const Route = createFileRoute("/results")({
  head: () => ({
    meta: [
      { title: "Results — Lumen AI Data Studio" },
      { name: "description", content: "Prediction history and exportable results." },
    ],
  }),
  component: Results,
});

const HISTORY = [
  { id: "p_4821", at: "2026-05-12 10:42", model: "churn-v3", label: "Will retain", confidence: 0.92 },
  { id: "p_4820", at: "2026-05-12 10:31", model: "churn-v3", label: "Will churn", confidence: 0.81 },
  { id: "p_4819", at: "2026-05-12 10:14", model: "churn-v3", label: "Will retain", confidence: 0.74 },
  { id: "p_4818", at: "2026-05-12 09:58", model: "churn-v3", label: "Will retain", confidence: 0.88 },
  { id: "p_4817", at: "2026-05-12 09:41", model: "churn-v3", label: "Will churn", confidence: 0.66 },
  { id: "p_4816", at: "2026-05-12 09:22", model: "churn-v3", label: "Will retain", confidence: 0.95 },
  { id: "p_4815", at: "2026-05-12 09:05", model: "churn-v3", label: "Will churn", confidence: 0.71 },
];

const BUCKETS = [
  { label: "0–20%", count: 2 },
  { label: "20–40%", count: 4 },
  { label: "40–60%", count: 7 },
  { label: "60–80%", count: 12 },
  { label: "80–100%", count: 18 },
];

function Results() {
  const max = Math.max(...BUCKETS.map((b) => b.count));

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Prediction history</h1>
          <p className="text-sm text-muted-foreground">43 predictions · last 7 days</p>
        </div>
        <Button variant="outline" className="gap-2">
          <Download className="h-4 w-4" /> Export CSV
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Confidence distribution</CardTitle>
          <CardDescription>Across all predictions in the current run</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex h-48 items-end gap-4">
            {BUCKETS.map((b) => (
              <div key={b.label} className="flex flex-1 flex-col items-center gap-2">
                <div className="flex w-full flex-1 items-end">
                  <div
                    className="w-full rounded-t-md transition-all"
                    style={{
                      height: `${(b.count / max) * 100}%`,
                      backgroundImage: "var(--gradient-primary)",
                      boxShadow: "var(--shadow-glow)",
                    }}
                  />
                </div>
                <div className="text-xs text-muted-foreground">{b.label}</div>
                <div className="font-mono text-xs">{b.count}</div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recent predictions</CardTitle>
          <CardDescription>Click a row to inspect inputs and explanation.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto rounded-md border border-border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs uppercase tracking-wider">ID</TableHead>
                  <TableHead className="text-xs uppercase tracking-wider">Time</TableHead>
                  <TableHead className="text-xs uppercase tracking-wider">Model</TableHead>
                  <TableHead className="text-xs uppercase tracking-wider">Prediction</TableHead>
                  <TableHead className="text-right text-xs uppercase tracking-wider">Confidence</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {HISTORY.map((h) => {
                  const positive = h.label === "Will retain";
                  return (
                    <TableRow key={h.id}>
                      <TableCell className="font-mono text-xs">{h.id}</TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">{h.at}</TableCell>
                      <TableCell className="font-mono text-xs">{h.model}</TableCell>
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={`gap-1 ${positive ? "border-success/40 text-success" : "border-destructive/40 text-destructive"}`}
                        >
                          {positive ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                          {h.label}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="ml-auto flex w-32 items-center gap-2">
                          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                            <div
                              className="h-full rounded-full"
                              style={{ width: `${h.confidence * 100}%`, backgroundImage: "var(--gradient-primary)" }}
                            />
                          </div>
                          <span className="font-mono text-xs">{(h.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}