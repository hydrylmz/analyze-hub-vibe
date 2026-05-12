// src/lib/csv-store.ts
// Basit global store — Context veya Zustand gerekmez,
// modül seviyesinde tutulur, sayfa geçişlerinde korunur.

export type CsvStore = {
  fileName: string;
  fileSize: number;
  headers: string[];
  rows: string[][];        // tüm satırlar (ham)
  numericCols: string[];   // sadece sayısal kolonlar
  categoricalCols: string[]; // kategorik kolonlar
};

let _store: CsvStore | null = null;

export function setStore(s: CsvStore) {
  _store = s;
}

export function getStore(): CsvStore | null {
  return _store;
}

export function clearStore() {
  _store = null;
}

// Bir kolondaki değerlerin sayısal olup olmadığını kontrol et
export function isNumericCol(rows: string[][], colIdx: number): boolean {
  const sample = rows.slice(0, 20).map((r) => r[colIdx]).filter(Boolean);
  return sample.length > 0 && sample.every((v) => !isNaN(Number(v.replace(/,/g, ""))));
}

// Kolon istatistikleri
export function colStats(rows: string[][], colIdx: number) {
  const vals = rows
    .map((r) => Number(r[colIdx]?.replace(/,/g, "")))
    .filter((v) => !isNaN(v));
  if (vals.length === 0) return null;
  const sorted = [...vals].sort((a, b) => a - b);
  const sum = vals.reduce((a, b) => a + b, 0);
  return {
    min: sorted[0],
    max: sorted[sorted.length - 1],
    mean: sum / vals.length,
    median: sorted[Math.floor(sorted.length / 2)],
    count: vals.length,
  };
}

// Histogram için bucket sayımları (8 bin)
export function histogram(rows: string[][], colIdx: number, bins = 8): number[] {
  const stats = colStats(rows, colIdx);
  if (!stats) return Array(bins).fill(0);
  const { min, max } = stats;
  const range = max - min || 1;
  const counts = Array(bins).fill(0);
  rows.forEach((r) => {
    const v = Number(r[colIdx]?.replace(/,/g, ""));
    if (isNaN(v)) return;
    const idx = Math.min(Math.floor(((v - min) / range) * bins), bins - 1);
    counts[idx]++;
  });
  return counts;
}

// Pearson korelasyon (-1..1)
export function pearson(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n === 0) return 0;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  const num = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0);
  const dx = Math.sqrt(xs.reduce((s, x) => s + (x - mx) ** 2, 0));
  const dy = Math.sqrt(ys.reduce((s, y) => s + (y - my) ** 2, 0));
  if (dx === 0 || dy === 0) return 0;
  return Math.round((num / (dx * dy)) * 100) / 100;
}

// Korelasyon matrisi (sadece sayısal kolonlar)
export function corrMatrix(rows: string[][], numCols: string[], headers: string[]): number[][] {
  const colData = numCols.map((col) => {
    const idx = headers.indexOf(col);
    return rows.map((r) => Number(r[idx]?.replace(/,/g, ""))).filter((v) => !isNaN(v));
  });
  return colData.map((xs, i) =>
    colData.map((ys, j) => {
      if (i === j) return 1;
      const len = Math.min(xs.length, ys.length);
      return pearson(xs.slice(0, len), ys.slice(0, len));
    })
  );
}

// Eksik değer sayısı
export function missingCount(rows: string[][], headers: string[]): Record<string, number> {
  const counts: Record<string, number> = {};
  headers.forEach((h, i) => {
    counts[h] = rows.filter((r) => !r[i] || r[i].trim() === "").length;
  });
  return counts;
}