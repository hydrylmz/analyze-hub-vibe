// src/routes/batch.tsx
import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useState } from "react";
import {
  Upload,
  FileSpreadsheet,
  Download,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  X,
  Loader2,
  Settings2,
  Trophy,
} from "lucide-react";
import { Button } from "@/components/ui/button";
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
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import Papa from "papaparse";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export const Route = createFileRoute("/batch")({
  head: () => ({
    meta: [
      { title: "Batch Predict — Lumen AI Data Studio" },
      { name: "description", content: "Upload a CSV and predict all rows at once." },
    ],
  }),
  component: BatchPredict,
});

type PredictionRow = {
  prediction: string | number;
  confidence: number | null;
};

type ExportMode = "full" | "submission" | "custom";

function BatchPredict() {
  const [file, setFile] = useState<{ name: string; size: number } | null>(null);
  const [headers, setHeaders] = useState<string[]>([]);
  const [rows, setRows] = useState<Record<string, string>[]>([]);
  const [previewRows, setPreviewRows] = useState<Record<string, string>[]>([]);
  const [predictions, setPredictions] = useState<PredictionRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  // Export state
  const [exportMode, setExportMode] = useState<ExportMode>("full");
  const [selectedCols, setSelectedCols] = useState<string[]>([]);
  const [includeConfidence, setIncludeConfidence] = useState(true);
  const [submissionIdCol, setSubmissionIdCol] = useState<string>("");
  const [submissionPredLabel, setSubmissionPredLabel] = useState("SalePrice");

  const parseFile = useCallback((f: File) => {
    setError(null);
    setDone(false);
    setPredictions([]);
    setFile({ name: f.name, size: f.size });

    Papa.parse<Record<string, string>>(f, {
      header: true,
      skipEmptyLines: true,
      complete: (result) => {
        if (!result.data.length) {
          setError("CSV is empty or has no header row.");
          return;
        }
        const hdrs = Object.keys(result.data[0]);
        setHeaders(hdrs);
        setRows(result.data);
        setPreviewRows(result.data.slice(0, 5));
        setSelectedCols(hdrs);
        const idCol = hdrs.find((h) => h.toLowerCase() === "id");
        if (idCol) setSubmissionIdCol(idCol);
      },
      error: () => setError("Dosya okunamadı."),
    });
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer.files?.[0];
      if (f) parseFile(f);
    },
    [parseFile]
  );

  const onPredict = async () => {
    if (!rows.length) return;
    setLoading(true);
    setError(null);
    setDone(false);
    setPredictions([]);

    try {
      const res = await fetch(`${API_URL}/predict/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows }),
      });
      if (!res.ok) {
        const e = await res.json();
        throw new Error(e.detail ?? "Batch prediction error.");
      }
      const data = await res.json();
      setPredictions(data.predictions);
      setDone(true);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "No backend connection.");
    } finally {
      setLoading(false);
    }
  };

  const buildExportData = () => {
    if (exportMode === "submission") {
      return rows.map((row, i) => ({
        [submissionIdCol || "Id"]: row[submissionIdCol] ?? i + 1,
        [submissionPredLabel]: predictions[i]?.prediction ?? "",
      }));
    }
    if (exportMode === "custom") {
      return rows.map((row, i) => {
        const out: Record<string, unknown> = {};
        selectedCols.forEach((col) => { out[col] = row[col] ?? ""; });
        out["prediction"] = predictions[i]?.prediction ?? "";
        if (includeConfidence && predictions[i]?.confidence != null) {
          out["confidence"] = predictions[i].confidence;
        }
        return out;
      });
    }
    // full
    return rows.map((row, i) => ({
      ...row,
      prediction: predictions[i]?.prediction ?? "",
      ...(includeConfidence ? { confidence: predictions[i]?.confidence ?? "" } : {}),
    }));
  };

  const onDownload = () => {
    if (!predictions.length) return;
    const data = buildExportData();
    const csv = Papa.unparse(data);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const suffix = exportMode === "submission" ? "submission" : exportMode === "custom" ? "custom" : "full";
    a.download = `${suffix}_${file?.name ?? "output"}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const toggleCol = (col: string) => {
    setSelectedCols((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const onRemove = () => {
    setFile(null);
    setHeaders([]);
    setRows([]);
    setPreviewRows([]);
    setPredictions([]);
    setDone(false);
    setError(null);
  };

  const displayHeaders = done ? [...headers, "prediction", "confidence"] : headers;
  const displayRows = done
    ? previewRows.map((row, i) => ({
        ...row,
        prediction: String(predictions[i]?.prediction ?? ""),
        confidence:
          predictions[i]?.confidence != null
            ? `${(predictions[i].confidence * 100).toFixed(1)}%`
            : "—",
      }))
    : previewRows;

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Batch Prediction</h1>
        <p className="text-sm text-muted-foreground">
          Upload CSV → generate predictions for all rows → configure output → download.
        </p>
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          <AlertCircle className="h-4 w-4 shrink-0" />
          {error}
        </div>
      )}

      {/* Upload */}
      {!file && (
        <>
          <Card>
            <CardContent className="p-6">
              <label
                onDragOver={(e) => e.preventDefault()}
                onDrop={onDrop}
                className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border bg-muted/30 px-6 py-16 text-center transition-colors hover:border-primary/60"
              >
                <input
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) parseFile(f);
                  }}
                />
                <div
                  className="mb-4 flex h-14 w-14 items-center justify-center rounded-full text-primary-foreground"
                  style={{ backgroundImage: "var(--gradient-primary)", boxShadow: "var(--shadow-glow)" }}
                >
                  <Upload className="h-6 w-6" />
                </div>
                <p className="text-base font-medium">Upload CSV for prediction</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  Must include the columns you used in training · without the target column · max 200MB
                </p>
              </label>
            </CardContent>
          </Card>

          <Card className="border-dashed">
            <CardContent className="p-6">
              <p className="mb-3 text-sm font-medium">How it works?</p>
              <ol className="space-y-2 text-sm text-muted-foreground list-decimal list-inside">
                <li>First, <strong className="text-foreground">train the model at </strong> page.</li>
                <li>Upload a CSV without the target column.</li>
                <li><strong className="text-foreground">Guess</strong> → select export output → download.</li>
              </ol>
            </CardContent>
          </Card>
        </>
      )}

      {/* Preview table */}
      {file && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-4 space-y-0">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-md bg-primary/10 text-primary">
                <FileSpreadsheet className="h-4 w-4" />
              </div>
              <div>
                <CardTitle className="text-base">{file.name}</CardTitle>
                <CardDescription>
                  {(file.size / 1024).toFixed(1)} KB · {rows.length.toLocaleString()} rows ·{" "}
                  {headers.length} columns · first 5 rows preview
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {done && (
                <Badge variant="secondary" className="gap-1 text-success">
                  <CheckCircle2 className="h-3 w-3" />
                  {predictions.length} predictions ready
                </Badge>
              )}
              <Button variant="ghost" size="icon" onClick={onRemove}>
                <X className="h-4 w-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="overflow-x-auto rounded-md border border-border">
              <Table>
                <TableHeader>
                  <TableRow>
                    {displayHeaders.map((h) => (
                      <TableHead
                        key={h}
                        className={`font-mono text-xs uppercase tracking-wider ${
                          h === "prediction" ? "text-primary" : h === "confidence" ? "text-muted-foreground" : ""
                        }`}
                      >
                        {h}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {displayRows.map((row, i) => (
                    <TableRow key={i}>
                      {displayHeaders.map((h) => (
                        <TableCell
                          key={h}
                          className={`font-mono text-xs ${h === "prediction" ? "font-semibold text-primary" : ""}`}
                        >
                          {row[h] ?? "—"}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            {rows.length > 5 && (
              <p className="text-center text-xs text-muted-foreground">
                +{rows.length - 5} rows more · after prediction, you can download all rows with predictions in a CSV file.
              </p>
            )}
            <div className="flex justify-end">
              <Button onClick={onPredict} disabled={loading} className="gap-2">
                {loading ? (
                  <><Loader2 className="h-4 w-4 animate-spin" />Predicting… ({rows.length} rows)</>
                ) : (
                  <><Sparkles className="h-4 w-4" />{done ? "Predict Again" : `Predict (${rows.length} rows)`}</>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Export config */}
      {done && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Settings2 className="h-4 w-4 text-primary" />
              Output Configuration
            </CardTitle>
            <CardDescription>Configure the content and format of the file you will download.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Mode selector */}
            <div className="grid gap-3 sm:grid-cols-3">
              <button
                onClick={() => setExportMode("full")}
                className={`rounded-lg border p-4 text-left transition-colors ${
                  exportMode === "full" ? "border-primary bg-primary/5" : "border-border hover:border-primary/40"
                }`}
              >
                <div className="mb-1 flex items-center gap-2">
                  <FileSpreadsheet className="h-4 w-4 text-primary" />
                  <span className="text-sm font-medium">Full Data</span>
                </div>
                <p className="text-xs text-muted-foreground">
                  All original columns + prediction + confidence
                </p>
              </button>

              <button
                onClick={() => setExportMode("submission")}
                className={`rounded-lg border p-4 text-left transition-colors ${
                  exportMode === "submission" ? "border-primary bg-primary/5" : "border-border hover:border-primary/40"
                }`}
              >
                <div className="mb-1 flex items-center gap-2">
                  <Trophy className="h-4 w-4 text-primary" />
                  <span className="text-sm font-medium">Kaggle Submission</span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Only Id + prediction columns, with configurable column names for Kaggle submissions
                </p>
              </button>

              <button
                onClick={() => setExportMode("custom")}
                className={`rounded-lg border p-4 text-left transition-colors ${
                  exportMode === "custom" ? "border-primary bg-primary/5" : "border-border hover:border-primary/40"
                }`}
              >
                <div className="mb-1 flex items-center gap-2">
                  <Settings2 className="h-4 w-4 text-primary" />
                  <span className="text-sm font-medium">Özel Seçim</span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Choose which original columns to include + prediction + confidence
                </p>
              </button>
            </div>

            {/* Submission options */}
            {exportMode === "submission" && (
              <div className="rounded-md border border-border bg-muted/30 p-4 space-y-3">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <Label className="text-xs uppercase tracking-wider text-muted-foreground">
                      ID Colon
                    </Label>
                    <select
                      value={submissionIdCol}
                      onChange={(e) => setSubmissionIdCol(e.target.value)}
                      className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                    >
                      {headers.map((h) => (
                        <option key={h} value={h}>{h}</option>
                      ))}
                    </select>
                  </div>
                  <div className="space-y-2">
                    <Label className="text-xs uppercase tracking-wider text-muted-foreground">
                      Prediction Colon Name
                    </Label>
                    <input
                      value={submissionPredLabel}
                      onChange={(e) => setSubmissionPredLabel(e.target.value)}
                      className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
                      placeholder="SalePrice"
                    />
                  </div>
                </div>
                <p className="text-xs text-muted-foreground font-mono">
                  Preview: {submissionIdCol || "Id"},{submissionPredLabel}
                  <br />1461,125341.779
                </p>
              </div>
            )}

            {/* Custom column selector */}
            {exportMode === "custom" && (
              <div className="rounded-md border border-border bg-muted/30 p-4 space-y-3">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-sm font-medium">Colons To Include</p>
                  <div className="flex gap-2">
                    <button onClick={() => setSelectedCols([...headers])} className="text-xs text-primary hover:underline">
                      Tümünü seç
                    </button>
                    <span className="text-muted-foreground">·</span>
                    <button onClick={() => setSelectedCols([])} className="text-xs text-muted-foreground hover:underline">
                      Temizle
                    </button>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 max-h-48 overflow-y-auto">
                  {headers.map((col) => (
                    <div key={col} className="flex items-center gap-2">
                      <Checkbox
                        id={`col-${col}`}
                        checked={selectedCols.includes(col)}
                        onCheckedChange={() => toggleCol(col)}
                      />
                      <label htmlFor={`col-${col}`} className="text-xs font-mono cursor-pointer truncate">
                        {col}
                      </label>
                    </div>
                  ))}
                </div>
                <div className="flex items-center gap-2 pt-1 border-t border-border">
                  <Checkbox
                    id="include-confidence"
                    checked={includeConfidence}
                    onCheckedChange={(v) => setIncludeConfidence(!!v)}
                  />
                  <label htmlFor="include-confidence" className="text-xs cursor-pointer">
                    Include confidence scores in the output CSV
                  </label>
                </div>
              </div>
            )}

            {/* Full mode confidence toggle */}
            {exportMode === "full" && (
              <div className="flex items-center gap-2">
                <Checkbox
                  id="full-confidence"
                  checked={includeConfidence}
                  onCheckedChange={(v) => setIncludeConfidence(!!v)}
                />
                <label htmlFor="full-confidence" className="text-sm cursor-pointer">
                  Include confidence scores in the output CSV
                </label>
              </div>
            )}

            <div className="flex justify-end">
              <Button onClick={onDownload} className="gap-2">
                <Download className="h-4 w-4" />
                {exportMode === "submission"
                  ? "Download Submission CSV"
                  : exportMode === "custom"
                  ? `Download Special CSV (${selectedCols.length} kolon)`
                  : `Download Full CSV (${predictions.length} satır)`}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}