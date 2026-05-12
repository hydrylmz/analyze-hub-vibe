import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { Download, TrendingUp, TrendingDown, RefreshCw, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export const Route = createFileRoute("/results")({
  head: () => ({
    meta: [
      { title: "Results — Lumen AI Data Studio" },
      { name: "description", content: "Prediction history and exportable results." },
    ],
  }),
  component: Results,
});

type HistoryItem = {
  id: string;
  at: string;
  model: string;
  target: string;
  prediction: string | number;
  confidence: number | null;
  inputs: Record<string, string | number>;
};

type ResultsData = {
  total: number;
  history: HistoryItem[];
  metrics: Record<string, unknown> | null;
  task: string | null;
  target: string | null;
  confidence_buckets: { label: string; count: number }[];
};

function Results() {
  const navigate = useNavigate();
  const [data, setData] = useState<ResultsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchResults = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/results`);
      if (!res.ok) {
        const e = await res.json();
        throw new Error(e.detail ?? "Sonuç alınamadı");
      }
      setData(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Backend bağlantısı yok.");
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = async () => {
    await fetch(`${API_URL}/results`, { method: "DELETE" });
    fetchResults();
  };

  const exportCSV = () => {
    if (!data?.history.length) return;
    const headers = ["id", "at", "model", "target", "prediction", "confidence"];
    const rows = data.history.map((h) =>
      [h.id, h.at, h.model, h.target, h.prediction, h.confidence ?? ""].join(",")
    );
    const csv = [headers.join(","), ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => { fetchResults(); }, []);

  if (loading) {
    return (
      <div className="mx-auto max-w-6xl p-6">
        <p className="text-sm text-muted-foreground">Yükleniyor…</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="mx-auto max-w-6xl space-y-4 p-6">
        <h1 className="text-2xl font-semibold tracking-tight">Prediction history</h1>
        <p className="text-sm text-muted-foreground">
          {error ?? "Henüz tahmin yapılmadı."}
        </p>
        <Button onClick={() => navigate({ to: "/predict" })}>← Predict sayfasına git</Button>
      </div>
    );
  }

  const maxBucket = Math.max(...data.confidence_buckets.map((b) => b.count), 1);
  const hasConfidence = data.task === "classification";

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Prediction history</h1>
          <p className="text-sm text-muted-foreground">
            {data.total} prediction{data.total !== 1 ? "s" : ""}
            {data.target ? ` · target: ${data.target}` : ""}
            {data.task ? ` · ${data.task}` : ""}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="gap-2" onClick={fetchResults}>
            <RefreshCw className="h-4 w-4" /> Refresh
          </Button>
          <Button variant="outline" size="sm" className="gap-2" onClick={exportCSV} disabled={!data.history.length}>
            <Download className="h-4 w-4" /> Export CSV
          </Button>
          <Button variant="outline" size="sm" className="gap-2 text-destructive hover:text-destructive" onClick={clearHistory} disabled={!data.history.length}>
            <Trash2 className="h-4 w-4" /> Clear
          </Button>
        </div>
      </div>

      {/* Model metrikleri */}
      {data.metrics && (
        <Card>
          <CardHeader>
            <CardTitle>Model metrics</CardTitle>
            <CardDescription>Son eğitim sonuçları</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-6 font-mono text-sm">
              {Object.entries(data.metrics).map(([k, v]) => (
                <div key={k}>
                  <p className="text-xs uppercase tracking-wider text-muted-foreground">{k}</p>
                  <p className="text-xl font-semibold text-foreground">{String(v)}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Confidence dağılımı — sadece classification'da göster */}
      {hasConfidence && data.confidence_buckets.some((b) => b.count > 0) && (
        <Card>
          <CardHeader>
            <CardTitle>Confidence distribution</CardTitle>
            <CardDescription>Tüm tahminlerdeki güven skoru dağılımı</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex h-48 items-end gap-4">
              {data.confidence_buckets.map((b) => (
                <div key={b.label} className="flex flex-1 flex-col items-center gap-2">
                  <div className="flex w-full flex-1 items-end">
                    <div
                      className="w-full rounded-t-md transition-all"
                      style={{
                        height: `${(b.count / maxBucket) * 100}%`,
                        backgroundImage: "var(--gradient-primary)",
                        boxShadow: b.count > 0 ? "var(--shadow-glow)" : undefined,
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
      )}

      {/* Tahmin geçmişi tablosu */}
      <Card>
        <CardHeader>
          <CardTitle>Recent predictions</CardTitle>
          <CardDescription>Son {data.history.length} tahmin</CardDescription>
        </CardHeader>
        <CardContent>
          {data.history.length === 0 ? (
            <p className="text-sm text-muted-foreground">Henüz tahmin yok.</p>
          ) : (
            <div className="overflow-x-auto rounded-md border border-border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="text-xs uppercase tracking-wider">ID</TableHead>
                    <TableHead className="text-xs uppercase tracking-wider">Time</TableHead>
                    <TableHead className="text-xs uppercase tracking-wider">Target</TableHead>
                    <TableHead className="text-xs uppercase tracking-wider">Prediction</TableHead>
                    {hasConfidence && (
                      <TableHead className="text-right text-xs uppercase tracking-wider">Confidence</TableHead>
                    )}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.history.map((h) => {
                    const isPositive =
                      typeof h.prediction === "string" &&
                      (h.prediction.toLowerCase().includes("retain") ||
                        h.prediction.toLowerCase().includes("pass") ||
                        h.prediction.toLowerCase().includes("yes") ||
                        h.prediction.toLowerCase().includes("true"));

                    return (
                      <TableRow key={h.id}>
                        <TableCell className="font-mono text-xs">{h.id}</TableCell>
                        <TableCell className="font-mono text-xs text-muted-foreground">{h.at}</TableCell>
                        <TableCell className="font-mono text-xs">{h.target}</TableCell>
                        <TableCell>
                          {data.task === "classification" ? (
                            <Badge
                              variant="outline"
                              className={`gap-1 ${isPositive ? "border-success/40 text-success" : "border-destructive/40 text-destructive"}`}
                            >
                              {isPositive
                                ? <TrendingUp className="h-3 w-3" />
                                : <TrendingDown className="h-3 w-3" />}
                              {String(h.prediction)}
                            </Badge>
                          ) : (
                            <span className="font-mono text-sm font-semibold">
                              {typeof h.prediction === "number"
                                ? h.prediction.toFixed(2)
                                : h.prediction}
                            </span>
                          )}
                        </TableCell>
                        {hasConfidence && (
                          <TableCell className="text-right">
                            {h.confidence !== null ? (
                              <div className="ml-auto flex w-32 items-center gap-2">
                                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                                  <div
                                    className="h-full rounded-full"
                                    style={{
                                      width: `${(h.confidence ?? 0) * 100}%`,
                                      backgroundImage: "var(--gradient-primary)",
                                    }}
                                  />
                                </div>
                                <span className="font-mono text-xs">
                                  {((h.confidence ?? 0) * 100).toFixed(0)}%
                                </span>
                              </div>
                            ) : "—"}
                          </TableCell>
                        )}
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}