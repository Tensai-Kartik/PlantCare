import React, { useState, useEffect } from 'react';
import { Sun, Moon, Plus, RotateCcw, Download, Printer } from 'lucide-react';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';
import { Dashboard } from './pages/Dashboard';
import { AnalyzePlant } from './pages/AnalyzePlant';
import { KnowledgeBase } from './pages/KnowledgeBase';
import { CommonDiseases } from './pages/CommonDiseases';
import { TreatmentGuide } from './pages/TreatmentGuide';
import { TipsPrevention } from './pages/TipsPrevention';
import { QualityModal } from './components/quality/QualityModal';
import { AnalyzingScreen } from './components/analysis/AnalyzingScreen';
import { ResultView } from './components/results/ResultView';
import { 
  PageRoute, 
  ThemeMode, 
  ModelMetadata, 
  ExampleLeaf, 
  QualityCheckResult, 
  AnalysisResponse 
} from './types';
import { 
  fetchModels, 
  fetchExamples, 
  checkImageQuality, 
  analyzePlant, 
  analyzeExample 
} from './services/api';
import { exportAnalysisAsJSON, printDiagnosticReport } from './utils/reportGenerator';

export function App() {
  const [currentPage, setCurrentPage] = useState<PageRoute>('dashboard');
  const [theme, setTheme] = useState<ThemeMode>(() => {
    return (localStorage.getItem('plantcare_theme') as ThemeMode) || 'light';
  });

  const [availableModels, setAvailableModels] = useState<ModelMetadata[]>([
    {
      id: 'efficientnet_b0',
      name: 'EfficientNet-B0',
      architecture: 'efficientnet_b0',
      accuracy: 90.48,
      weighted_f1: 0.8845,
      latency_ms: 13.03,
      is_default: true
    },
    {
      id: 'mobilenet_v3_small',
      name: 'MobileNetV3-Small',
      architecture: 'mobilenet_v3_small',
      accuracy: 77.78,
      weighted_f1: 0.7135,
      latency_ms: 1.91,
      is_default: false
    }
  ]);
  const [selectedModel, setSelectedModel] = useState<string>('efficientnet_b0');
  const [examples, setExamples] = useState<ExampleLeaf[]>([]);
  const [selectedDiseaseForKB, setSelectedDiseaseForKB] = useState<string | null>(null);

  // Direct Analysis State triggered from Dashboard or Analyze page
  const [directAnalysisStep, setDirectAnalysisStep] = useState<'idle' | 'quality' | 'analyzing' | 'result'>('idle');
  const [directFile, setDirectFile] = useState<File | null>(null);
  const [directPreviewUrl, setDirectPreviewUrl] = useState<string | null>(null);
  const [directQuality, setDirectQuality] = useState<QualityCheckResult | null>(null);
  const [directResult, setDirectResult] = useState<AnalysisResponse | null>(null);
  const [isGlobalLoading, setIsGlobalLoading] = useState<boolean>(false);

  useEffect(() => {
    // Apply theme to document root
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('plantcare_theme', theme);
  }, [theme]);

  useEffect(() => {
    // Load models and examples
    fetchModels().then((res) => {
      if (res.models && res.models.length > 0) {
        setAvailableModels(res.models);
        setSelectedModel(res.default || res.models[0].id);
      }
    });

    fetchExamples().then((data) => {
      setExamples(data);
    });
  }, []);

  const handleToggleTheme = () => {
    setTheme(prev => (prev === 'light' ? 'dark' : 'light'));
  };

  const handleDashboardImageSelected = async (file: File) => {
    setDirectFile(file);
    const url = URL.createObjectURL(file);
    setDirectPreviewUrl(url);
    setIsGlobalLoading(true);

    try {
      const qRes = await checkImageQuality(file);
      setDirectQuality(qRes);
      setDirectAnalysisStep('quality');
    } catch (err) {
      console.error('Quality check failed:', err);
      // Fallback straight to analysis
      executeDirectAnalysis(file, true);
    } finally {
      setIsGlobalLoading(false);
    }
  };

  const executeDirectAnalysis = async (fileToUse?: File, skipQuality: boolean = false) => {
    const file = fileToUse || directFile;
    if (!file) return;

    setDirectAnalysisStep('analyzing');
    setIsGlobalLoading(true);

    try {
      const res = await analyzePlant(file, selectedModel, skipQuality, true);
      setDirectResult(res);
      setDirectAnalysisStep('result');
    } catch (err) {
      console.error('Analysis error:', err);
      alert('Analysis encountered an issue. Please try another image.');
      handleResetDirect();
    } finally {
      setIsGlobalLoading(false);
    }
  };

  const handleSelectExample = async (exampleId: string) => {
    setIsGlobalLoading(true);
    const exObj = examples.find(e => e.id === exampleId);
    const previewUrl = exObj?.image_url || `/examples/${exampleId}.jpg`;
    setDirectPreviewUrl(previewUrl);
    setDirectAnalysisStep('analyzing');

    try {
      const res = await analyzeExample(exampleId, selectedModel, true);
      setDirectResult(res);
      setDirectAnalysisStep('result');
    } catch (err) {
      console.error('Example analysis error:', err);
      alert('Failed to analyze sample. Please ensure backend is running.');
      handleResetDirect();
    } finally {
      setIsGlobalLoading(false);
    }
  };

  const handleResetDirect = () => {
    setDirectAnalysisStep('idle');
    setDirectFile(null);
    if (directPreviewUrl && !directPreviewUrl.startsWith('/examples')) {
      URL.revokeObjectURL(directPreviewUrl);
    }
    setDirectPreviewUrl(null);
    setDirectQuality(null);
    setDirectResult(null);
  };

  const handleNavigate = (page: PageRoute) => {
    handleResetDirect();
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleOpenDiseaseInKB = (diseaseId: string) => {
    setSelectedDiseaseForKB(diseaseId);
    handleResetDirect();
    setCurrentPage('knowledge');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const currentModelName = availableModels.find(m => m.id === selectedModel)?.name || 'EfficientNet-B0';

  const getPageTitle = (page: PageRoute) => {
    switch (page) {
      case 'dashboard': return 'Dashboard Overview';
      case 'analyze': return 'AI Plant Health Diagnosis';
      case 'knowledge': return 'Pathology Knowledge Base';
      case 'diseases': return 'Common Agricultural Diseases';
      case 'treatment': return 'Integrated Disease Treatment Guide';
      case 'prevention': return 'Cultural Best Practices & Prevention';
      default: return 'PlantCare';
    }
  };

  return (
    <div className="min-h-screen flex bg-primary text-primary-color font-sans antialiased">
      {/* Persistent Desktop Sidebar */}
      <div className="hidden md:block">
        <Sidebar
          currentPage={currentPage}
          onNavigate={handleNavigate}
          theme={theme}
          onToggleTheme={handleToggleTheme}
          selectedModel={selectedModel}
          availableModels={availableModels}
          onSelectModel={setSelectedModel}
        />
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile Header */}
        <Header
          currentPage={currentPage}
          onNavigate={handleNavigate}
          theme={theme}
          onToggleTheme={handleToggleTheme}
          onNewAnalysis={() => handleNavigate('analyze')}
          analysisResult={directAnalysisStep === 'result' ? directResult : null}
          onExportJSON={() => directResult && exportAnalysisAsJSON(directResult)}
          onDownloadReport={() => directResult && printDiagnosticReport(directResult)}
          onResetAnalysis={handleResetDirect}
        />

        {/* Desktop Top Header Bar */}
        <header className="hidden md:flex items-center justify-between px-8 py-3.5 border-b border-subtle bg-surface sticky top-0 z-30 select-none">
          {directAnalysisStep === 'result' && directResult ? (
            /* Post-Analysis Top Bar: All buttons in one line in exact sequence */
            <>
              <button
                onClick={handleResetDirect}
                className="flex items-center gap-2 text-xs font-semibold text-secondary-color hover:text-primary-color transition-colors cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>← Back to Dashboard</span>
              </button>

              {/* Action Buttons Sequence: Analyze another plant, dark/light switch, export to json, download report */}
              <div className="flex items-center gap-2">
                {/* 1. Analyze Another Plant */}
                <button
                  onClick={handleResetDirect}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold shadow-xs shadow-emerald-600/20 hover:scale-[1.02] active:scale-[0.98] transition-all cursor-pointer"
                  title="Analyze another plant"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>Analyze Another Plant</span>
                </button>

                {/* 2. Dark / Light switch */}
                <button
                  onClick={handleToggleTheme}
                  className="p-2 rounded-xl bg-surface-elevated hover:bg-surface text-secondary-color hover:text-primary-color border border-subtle transition-all cursor-pointer"
                  title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Mode`}
                >
                  {theme === 'light' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
                </button>

                {/* 3. Export to JSON */}
                <button
                  onClick={() => exportAnalysisAsJSON(directResult)}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-surface-elevated hover:bg-surface text-xs font-semibold text-secondary-color hover:text-primary-color border border-subtle transition-all cursor-pointer"
                  title="Download JSON metadata"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Export to JSON</span>
                </button>

                {/* 4. Download Report */}
                <button
                  onClick={() => printDiagnosticReport(directResult)}
                  className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-surface-elevated hover:bg-surface text-xs font-semibold text-primary-color border border-subtle transition-all cursor-pointer"
                  title="Print or export diagnostic report"
                >
                  <Printer className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                  <span>Download Report</span>
                </button>
              </div>
            </>
          ) : (
            /* Standard Pages Top Bar */
            <>
              <div className="flex items-center gap-2 text-sm font-semibold text-secondary-color">
                <span className="text-xs uppercase tracking-wider text-muted-color">PlantCare</span>
                <span className="text-muted-color">/</span>
                <span className="text-primary-color font-bold">
                  {getPageTitle(currentPage)}
                </span>
              </div>

              <div className="flex items-center gap-2.5">
                {currentPage !== 'analyze' && (
                  <button
                    onClick={() => handleNavigate('analyze')}
                    className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold shadow-xs shadow-emerald-600/20 hover:scale-[1.02] active:scale-[0.98] transition-all cursor-pointer"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    <span>Analyze Plant</span>
                  </button>
                )}

                {/* Dark/Light mode switch in top right corner */}
                <button
                  onClick={handleToggleTheme}
                  className="p-2 rounded-xl bg-surface-elevated hover:bg-surface text-secondary-color hover:text-primary-color border border-subtle transition-all cursor-pointer"
                  title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Mode`}
                >
                  {theme === 'light' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
                </button>
              </div>
            </>
          )}
        </header>

        {/* Page Container */}
        <main className="flex-1 p-4 sm:p-6 md:p-8 lg:p-10 max-w-7xl w-full mx-auto">
          {/* If Direct Analysis from Dashboard is active, render the step */}
          {directAnalysisStep === 'quality' && directPreviewUrl && directQuality ? (
            <div className="space-y-6">
              <QualityModal
                imagePreviewUrl={directPreviewUrl}
                qualityResult={directQuality}
                onProceed={() => executeDirectAnalysis(undefined, false)}
                onRetry={handleResetDirect}
              />
            </div>
          ) : directAnalysisStep === 'analyzing' && directPreviewUrl ? (
            <div className="space-y-6">
              <AnalyzingScreen
                imagePreviewUrl={directPreviewUrl}
                modelName={currentModelName}
              />
            </div>
          ) : directAnalysisStep === 'result' && directResult && directPreviewUrl ? (
            <ResultView
              analysis={directResult}
              originalImagePreview={directPreviewUrl}
              onReset={handleResetDirect}
              onOpenInKnowledgeBase={handleOpenDiseaseInKB}
            />
          ) : (
            <>
              {currentPage === 'dashboard' && (
                <Dashboard
                  onImageSelected={handleDashboardImageSelected}
                  onSelectExample={handleSelectExample}
                  examples={examples}
                  isLoading={isGlobalLoading}
                />
              )}

              {currentPage === 'analyze' && (
                <AnalyzePlant
                  selectedModel={selectedModel}
                  modelName={currentModelName}
                  onAnalysisComplete={(res, previewUrl) => {
                    setDirectResult(res);
                    setDirectPreviewUrl(previewUrl);
                    setDirectAnalysisStep('result');
                  }}
                />
              )}

              {currentPage === 'knowledge' && (
                <KnowledgeBase
                  initialDiseaseId={selectedDiseaseForKB}
                  onClearInitialDisease={() => setSelectedDiseaseForKB(null)}
                />
              )}

              {currentPage === 'diseases' && (
                <CommonDiseases onOpenDiseaseInKB={handleOpenDiseaseInKB} />
              )}

              {currentPage === 'treatment' && (
                <TreatmentGuide onOpenDiseaseInKB={handleOpenDiseaseInKB} />
              )}

              {currentPage === 'prevention' && <TipsPrevention />}
            </>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
