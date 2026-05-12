import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useState } from "react";
import { Link } from "@tanstack/react-router";
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

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Upload — Lumen AI Data Studio" },
      { name: "description", content: "Drop in a CSV to begin AI-powered exploratory analysis and prediction." },
    ],
  }),
  component: Index,
});

const SAMPLE_HEADERS = ["id", "age", "income", "score", "segment", "churned"];
const SAMPLE_ROWS = [
  ["1024", "34", "72,400", "812", "growth", "false"],
  ["1025", "47", "98,100", "766", "core", "false"],
  ["1026", "29", "41,300", "604", "trial", "true"],
  ["1027", "52", "120,900", "844", "core", "false"],
  ["1028", "41", "56,200", "688", "growth", "true"],
  ["1029", "38", "84,500", "791", "core", "false"],
];

function Index() {
  const [file, setFile] = useState<{ name: string; size: number } | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setFile({ name: f.name, size: f.size });
  }, []);

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
                if (f) setFile({ name: f.name, size: f.size });
              }}
            />
            <div
              className="mb-4 flex h-14 w-14 items-center justify-center rounded-full text-primary-foreground"
              style={{ backgroundImage: "var(--gradient-primary)", boxShadow: "var(--shadow-glow)" }}
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
        </CardContent>
      </Card>

      {(file || true) && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-4 space-y-0">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-md bg-primary/10 text-primary">
                <FileSpreadsheet className="h-4 w-4" />
              </div>
              <div>
                <CardTitle className="text-base">
                  {file?.name ?? "customers_sample.csv"}
                </CardTitle>
                <CardDescription>
                  {file ? `${(file.size / 1024).toFixed(1)} KB` : "12.4 KB"} · 6 columns ·
                  preview of first 6 rows
                </CardDescription>
              </div>
            </div>
            {file && (
              <Button variant="ghost" size="icon" onClick={() => setFile(null)}>
                <X className="h-4 w-4" />
              </Button>
            )}
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto rounded-md border border-border">
              <Table>
                <TableHeader>
                  <TableRow>
                    {SAMPLE_HEADERS.map((h) => (
                      <TableHead key={h} className="font-mono text-xs uppercase tracking-wider">
                        {h}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {SAMPLE_ROWS.map((r, i) => (
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
              <Button variant="outline">Cancel</Button>
              <Button asChild className="gap-2">
                <Link to="/eda">
                  Analyze <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
