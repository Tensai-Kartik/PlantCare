import { AnalysisResponse } from '../types';

export function printDiagnosticReport(analysis: AnalysisResponse) {
  window.print();
}

export function exportAnalysisAsJSON(analysis: AnalysisResponse) {
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(analysis, null, 2));
  const downloadAnchor = document.createElement('a');
  const filePrefix = analysis.prediction?.class_id || analysis.non_plant_details?.category || 'non_plant_inspection';
  downloadAnchor.setAttribute("href", dataStr);
  downloadAnchor.setAttribute("download", `PlantCare_Report_${filePrefix}_${Date.now()}.json`);
  document.body.appendChild(downloadAnchor);
  downloadAnchor.click();
  downloadAnchor.remove();
}
