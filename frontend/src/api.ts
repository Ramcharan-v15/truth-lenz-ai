import type { AnalysisReport } from './types';

const API = '/api';

async function check(res: Response) {
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const j = await res.json();
      if (j.detail) msg = j.detail;
    } catch { /* ignore */ }
    throw new Error(msg);
  }
  return res;
}

export async function analyzeImage(
  file: File,
  save: boolean,
  onProgress: (pct: number) => void,
): Promise<AnalysisReport> {
  // XHR for real upload progress
  const uploaded: Promise<void> = new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API}/analyze?save=${save ? 'true' : 'false'}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        (uploaded as any).result = JSON.parse(xhr.responseText);
        resolve();
      } else {
        let msg = `Analysis failed (${xhr.status})`;
        try { msg = JSON.parse(xhr.responseText).detail || msg; } catch { /* ignore */ }
        reject(new Error(msg));
      }
    };
    xhr.onerror = () => reject(new Error('Network error during upload.'));
    const fd = new FormData();
    fd.append('file', file);
    xhr.send(fd);
  });
  await uploaded;
  return (uploaded as any).result as AnalysisReport;
}

export async function compareImages(a: File, b: File) {
  const fd = new FormData();
  fd.append('file_a', a);
  fd.append('file_b', b);
  const res = await check(await fetch(`${API}/compare`, { method: 'POST', body: fd }));
  return res.json();
}

export async function batchAnalyze(files: File[]) {
  const fd = new FormData();
  files.forEach((f) => fd.append('files', f));
  const res = await check(await fetch(`${API}/batch`, { method: 'POST', body: fd }));
  return res.json();
}

export async function getHistory() {
  const res = await check(await fetch(`${API}/history`));
  return res.json();
}

export async function getAnalysis(id: string): Promise<AnalysisReport> {
  const res = await check(await fetch(`${API}/analysis/${id}`));
  return res.json();
}

export async function deleteAnalysis(id: string) {
  const res = await check(await fetch(`${API}/history/${id}`, { method: 'DELETE' }));
  return res.json();
}

export async function deleteAllHistory() {
  const res = await check(await fetch(`${API}/history`, { method: 'DELETE' }));
  return res.json();
}

export function reportUrl(id: string, format: 'json' | 'csv' | 'pdf') {
  return `${API}/report/${id}?format=${format}`;
}

export async function getConfig() {
  const res = await check(await fetch(`${API}/config`));
  return res.json();
}

export async function getHealth() {
  const res = await check(await fetch(`${API}/health`));
  return res.json();
}

// ---- V2 ----

export interface ModelRegistryStatus {
  models_dir: string;
  total: number;
  ready: number;
  models: {
    name: string;
    version: string;
    model_type: string;
    status: string;
    limitations: string;
    supported_formats: string[];
    preprocessing: Record<string, any>;
    lifecycle_status?: string;
    validation_status?: string;
    demonstration?: boolean;
    config?: {
      demonstration?: boolean;
      authorized_for_real_world_classification?: boolean;
    };
  }[];
  errors: Record<string, string>;
  ml_available: boolean;
}

export async function getModelStatus(signal?: AbortSignal): Promise<ModelRegistryStatus> {
  const res = await check(await fetch(`${API}/models`, signal ? { signal } : undefined));
  return res.json();
}

export async function reloadModels(): Promise<ModelRegistryStatus> {
  const res = await check(await fetch(`${API}/models/reload`, { method: 'POST' }));
  return res.json();
}

export async function getResearchView(id: string) {
  const res = await check(await fetch(`${API}/research/${id}`));
  return res.json();
}

export async function explainImage(file: File, level: 'basic' | 'technical' | 'forensic') {
  const fd = new FormData();
  fd.append('file', file);
  const res = await check(await fetch(`${API}/explain?level=${level}`, { method: 'POST', body: fd }));
  return res.json();
}

export async function runRobustness(file: File) {
  const fd = new FormData();
  fd.append('file', file);
  const res = await check(await fetch(`${API}/robustness`, { method: 'POST', body: fd }));
  return res.json();
}

export async function runBenchmark(dataset: string, version = '1.0') {
  const res = await check(await fetch(`${API}/benchmark/run?dataset=${encodeURIComponent(dataset)}&version=${encodeURIComponent(version)}`, { method: 'POST' }));
  return res.json();
}

export async function getBenchmarkResults(dataset: string) {
  const res = await check(await fetch(`${API}/benchmark/results?dataset=${encodeURIComponent(dataset)}`));
  return res.json();
}

export async function getBenchmarks() {
  const res = await check(await fetch(`${API}/benchmarks`));
  return res.json();
}

export async function getRobustnessInfo() {
  const res = await check(await fetch(`${API}/robustness`));
  return res.json();
}

// ---- V4 validation / research endpoints ----
export async function getValidation() {
  const res = await check(await fetch(`${API}/validation`));
  return res.json();
}

export async function getDatasets() {
  const res = await check(await fetch(`${API}/datasets`));
  return res.json();
}

export async function getExperiments() {
  const res = await check(await fetch(`${API}/experiments`));
  return res.json();
}

export async function getExperiment(id: string) {
  const res = await check(await fetch(`${API}/experiments/${encodeURIComponent(id)}`));
  return res.json();
}

export async function getCalibrationInfo() {
  const res = await check(await fetch(`${API}/calibration`));
  return res.json();
}

export async function getCrossGenerator() {
  const res = await check(await fetch(`${API}/cross-generator`));
  return res.json();
}

export async function getHardNegatives() {
  const res = await check(await fetch(`${API}/hard-negatives`));
  return res.json();
}

export async function getOodInfo() {
  const res = await check(await fetch(`${API}/ood`));
  return res.json();
}

// ---- V5 dataset registry / validation research endpoints ----
export async function getDatasetRegistry() {
  const res = await check(await fetch(`${API}/datasets/registry`));
  return res.json();
}

export async function getDataset(id: string) {
  const res = await check(await fetch(`${API}/datasets/registry/${encodeURIComponent(id)}`));
  return res.json();
}

export async function getDatasetCard(id: string): Promise<string> {
  const res = await check(await fetch(`${API}/datasets/registry/${encodeURIComponent(id)}/card`));
  return res.text();
}

export async function registerDataset(meta: { name: string; source?: string; license?: string; version?: string }) {
  const res = await check(await fetch(`${API}/datasets/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(meta),
  }));
  return res.json();
}

export async function getScorecard() {
  const res = await check(await fetch(`${API}/benchmark/scorecard`));
  return res.json();
}

export async function getFailures(experimentId: string) {
  const res = await check(await fetch(`${API}/benchmark/failures/${encodeURIComponent(experimentId)}`));
  return res.json();
}

export async function getGenerators(experimentId: string) {
  const res = await check(await fetch(`${API}/benchmark/generators/${encodeURIComponent(experimentId)}`));
  return res.json();
}

export async function getPromotion(modelId: string) {
  const res = await check(await fetch(`${API}/models/${encodeURIComponent(modelId)}/promotion`));
  return res.json();
}
