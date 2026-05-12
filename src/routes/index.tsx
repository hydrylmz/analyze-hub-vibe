// src/routes/index.tsx
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useCallback, useState } from "react";
import { Upload, FileSpreadsheet, X, ArrowRight } from "lucide-react";
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
import Papa from "papaparse";
import { setStore, isNumericCol, clearStore } from "@/lib/csv-store";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Upload — Lumen AI Data Studio" },
      {
        name: "description",
        content:
          "Drop in a CSV to begin AI-powered exploratory analysis and prediction.",
      },
    ],
  }),
  component: Index,
});

function Index() {
  const navigate = useNavigate();
  const [file, setFile] = useState<{ name: string; size: number } | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [headers, setHeaders] = useState<string[]>([]);
  const [rows, setRows] = useState<string[][]>([]);
  const [allRows, setAllRows] = useState<string[][]>([]);
  const [error, setError] = useState<string | null>(null);

  const parseFile = useCallback((f: File) => {
  setError(null);
  setFile({ name: f.name, size: f.size });

  // Backend'e gönder
  const formData = new FormData();
  formData.append("file", f);
  fetch("http://localhost:8000/upload", {
    method: "POST",
    body: formData,
  }).catch(() => console.warn("Backend bağlantısı yok, sadece frontend parse."));

  // Frontend parse (EDA için)
  Papa.parse<string[]>(f, {
    skipEmptyLines: true,
    complete: (result) => {
      const [hdrs, ...data] = result.data;
      if (!hdrs || hdrs.length === 0) {
        setError("CSV başlık satırı bulunamadı.");
        return;
      }
      const numericCols = hdrs.filter((_, i) => isNumericCol(data, i));
      const categoricalCols = hdrs.filter((h) => !numericCols.includes(h));
      setStore({
        fileName: f.name,
        fileSize: f.size,
        headers: hdrs,
        rows: data,
        numericCols,
        categoricalCols,
      });
      setHeaders(hdrs);
      setAllRows(data);
      setRows(data.slice(0, 6));
    },
    error: () => setError("Dosya okunamadı, geçerli bir CSV yükleyin."),
  });
}, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files?.[0];
      if (f) parseFile(f);
    },
    [parseFile]
  );

  const onRemove = () => {
    setFile(null);
    setHeaders([]);
    setRows([]);
    setAllRows([]);
    clearStore();
  };

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Upload dataset</h1>
        <p className="text-sm text-muted-foreground">
          Drop a CSV to profile, visualize, and run predictions.
        </p>
      </div>

      <Card>
        <CardContent className="p-6">
          <label
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-16 text-center transition-colors ${
              dragOver
                ? "border-primary bg-primary/5"
                : "border-border bg-muted/30 hover:border-primary/60"
            }`}
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
              style={{
                backgroundImage: "var(--gradient-primary)",
                boxShadow: "var(--shadow-glow)",
              }}
            >
              <Upload className="h-6 w-6" />
            </div>
            <p className="text-base font-medium">
              {dragOver ? "Release to upload" : "Drop your CSV here"}
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              or click to browse · max 200MB
            </p>
          </label>
          {error && (
            <p className="mt-3 text-center text-sm text-destructive">{error}</p>
          )}
        </CardContent>
      </Card>

      {file && headers.length > 0 && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-4 space-y-0">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-md bg-primary/10 text-primary">
                <FileSpreadsheet className="h-4 w-4" />
              </div>
              <div>
                <CardTitle className="text-base">{file.name}</CardTitle>
                <CardDescription>
                  {(file.size / 1024).toFixed(1)} KB · {headers.length} columns ·{" "}
                  {allRows.length.toLocaleString()} rows · preview of first 6
                </CardDescription>
              </div>
            </div>
            <Button variant="ghost" size="icon" onClick={onRemove}>
              <X className="h-4 w-4" />
            </Button>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto rounded-md border border-border">
              <Table>
                <TableHeader>
                  <TableRow>
                    {headers.map((h) => (
                      <TableHead
                        key={h}
                        className="font-mono text-xs uppercase tracking-wider"
                      >
                        {h}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.map((r, i) => (
                    <TableRow key={i}>
                      {r.map((c, j) => (
                        <TableCell key={j} className="font-mono text-xs">
                          {c}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            <div className="mt-6 flex items-center justify-end gap-3">
              <Button variant="outline" onClick={onRemove}>
                Cancel
              </Button>
              <Button
                className="gap-2"
                onClick={() => navigate({ to: "/eda" })}
              >
                Analyze <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}