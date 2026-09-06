import React, { useState, useEffect } from 'react';
import { 
  Menu, 
  X, 
  Sun, 
  Moon, 
  Leaf, 
  Plus, 
  RotateCcw, 
  Download, 
  Printer, 
  LayoutDashboard, 
  ScanLine, 
  BookOpen, 
  ShieldAlert, 
  Stethoscope, 
  Sprout, 
  Cpu, 
  CheckCircle2, 
  ChevronRight,
  Sparkles
} from 'lucide-react';
import { PageRoute, ThemeMode, ModelMetadata, AnalysisResponse } from '../../types';
import { subscribeServerStatus, ServerWarmupStatus } from '../../services/api';

interface HeaderProps {
  currentPage: PageRoute;
  onNavigate: (page: PageRoute) => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  onNewAnalysis: () => void;
  selectedModel?: string;
  availableModels?: ModelMetadata[];
  onSelectModel?: (modelId: string) => void;
  analysisResult?: AnalysisResponse | null;
  onExportJSON?: () => void;
  onDownloadReport?: () => void;
  onResetAnalysis?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentPage,
  onNavigate,
  theme,
  onToggleTheme,
  onNewAnalysis,
  selectedModel = 'efficientnet_b0',
  availableModels = [],
  onSelectModel,
  analysisResult,
  onExportJSON,
  onDownloadReport,
  onResetAnalysis
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const [serverStatus, setServerStatus] = useState<ServerWarmupStatus>('checking');

  useEffect(() => {
    const unsubscribe = subscribeServerStatus(setServerStatus);
    return () => unsubscribe();
  }, []);

  // Lock body scroll when mobile burger menu drawer is open
  useEffect(() => {
    if (mobileMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [mobileMenuOpen]);

  // Handle Escape key to close mobile drawer
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && mobileMenuOpen) {
        setMobileMenuOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [mobileMenuOpen]);

  const navItems: { id: PageRoute; label: string; icon: React.ReactNode; badge?: string }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard className="w-5 h-5" /> },
    { id: 'analyze', label: 'Analyze Plant', icon: <ScanLine className="w-5 h-5" />, badge: 'AI' },
    { id: 'knowledge', label: 'Knowledge Base', icon: <BookOpen className="w-5 h-5" /> },
    { id: 'diseases', label: 'Common Diseases', icon: <ShieldAlert className="w-5 h-5" /> },
    { id: 'treatment', label: 'Treatment Guide', icon: <Stethoscope className="w-5 h-5" /> },
    { id: 'prevention', label: 'Tips & Prevention', icon: <Sprout className="w-5 h-5" /> }
  ];

  const currentModelObj = availableModels.find(m => m.id === selectedModel) || availableModels[0];

  const handleNavClick = (page: PageRoute) => {
    onNavigate(page);
    setMobileMenuOpen(false);
  };

  return (
    <>
      {/* Mobile Top App Bar */}
      <header className="md:hidden bg-surface/95 backdrop-blur-md border-b border-subtle sticky top-0 z-40 px-3 sm:px-4 py-2.5 flex items-center justify-between transition-colors">
        {analysisResult && onResetAnalysis ? (
          /* Mobile Analysis Top Bar */
          <div className="w-full flex items-center justify-between gap-2">
            <button
              onClick={onResetAnalysis}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-surface-elevated text-xs font-semibold text-secondary-color hover:text-primary-color border border-subtle active:scale-95 transition-all cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Back</span>
            </button>

            <div className="flex items-center gap-1.5">
              <button
                onClick={onResetAnalysis}
                className="p-2 rounded-xl bg-emerald-600 text-white shadow-xs active:scale-95 transition-all cursor-pointer"
                title="Analyze Another Plant"
                aria-label="Analyze another plant"
              >
                <Plus className="w-4 h-4" />
              </button>

              <button
                onClick={onToggleTheme}
                className="p-2 rounded-xl bg-surface-elevated text-secondary-color border border-subtle active:scale-95 transition-all cursor-pointer"
                title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Mode`}
                aria-label="Toggle theme"
              >
                {theme === 'light' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
              </button>

              {onExportJSON && (
                <button
                  onClick={onExportJSON}
                  className="p-2 rounded-xl bg-surface-elevated text-secondary-color border border-subtle active:scale-95 transition-all cursor-pointer"
                  title="Export to JSON"
                  aria-label="Export JSON"
                >
                  <Download className="w-4 h-4" />
                </button>
              )}

              {onDownloadReport && (
                <button
                  onClick={onDownloadReport}
                  className="p-2 rounded-xl bg-surface-elevated text-primary-color border border-subtle active:scale-95 transition-all cursor-pointer"
                  title="Download Report"
                  aria-label="Download report"
                >
                  <Printer className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                </button>
              )}

              {/* Hamburger Button */}
              <button
                onClick={() => setMobileMenuOpen(true)}
                className="p-2 rounded-xl bg-surface-elevated text-primary-color border border-subtle active:scale-95 transition-all cursor-pointer ml-1"
                aria-label="Open navigation menu"
              >
                <Menu className="w-4 h-4" />
              </button>
            </div>
          </div>
        ) : (
          /* Standard Mobile Brand + Actions Header */
          <>
            <div 
              onClick={() => handleNavClick('dashboard')}
              className="flex items-center gap-2.5 cursor-pointer select-none"
            >
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-emerald-700 to-emerald-500 text-white flex items-center justify-center shadow-xs">
                <Leaf className="w-4 h-4" />
              </div>
              <div>
                <div className="flex items-center gap-1.5">
                  <span className="font-extrabold text-sm sm:text-base text-primary-color block leading-none">
                    PlantCare
                  </span>
                  <span className="relative flex h-2 w-2">
                    {serverStatus === 'online' && (
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                    )}
                    {serverStatus === 'waking' && (
                      <span className="animate-ping relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
                    )}
                    {serverStatus === 'checking' && (
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500 animate-pulse"></span>
                    )}
                    {serverStatus === 'offline' && (
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-zinc-400"></span>
                    )}
                  </span>
                </div>
                <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-medium leading-none block mt-0.5">
                  {serverStatus === 'waking' ? 'Warming up...' : 'AI Plant Health'}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-1.5 sm:gap-2">
              {currentPage !== 'analyze' && (
                <button
                  onClick={() => {
                    onNewAnalysis();
                    setMobileMenuOpen(false);
                  }}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold shadow-xs active:scale-95 transition-all cursor-pointer"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>Analyze</span>
                </button>
              )}

              <button
                onClick={onToggleTheme}
                className="p-2 rounded-xl bg-surface-elevated text-secondary-color hover:text-primary-color border border-subtle active:scale-95 transition-all cursor-pointer"
                title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Mode`}
                aria-label="Toggle theme"
              >
                {theme === 'light' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
              </button>

              {/* Hamburger Button */}
              <button
                onClick={() => setMobileMenuOpen(true)}
                className="p-2 rounded-xl bg-surface-elevated text-primary-color border border-subtle active:scale-95 transition-all cursor-pointer"
                aria-label="Open navigation menu"
              >
                <Menu className="w-5 h-5" />
              </button>
            </div>
          </>
        )}
      </header>

      {/* Mobile Drawer Menu Modal */}
      {mobileMenuOpen && (
        <div className="md:hidden fixed inset-0 z-50 flex">
          {/* Backdrop Overlay */}
          <div 
            onClick={() => setMobileMenuOpen(false)}
            className="fixed inset-0 bg-black/60 backdrop-blur-xs transition-opacity animate-in fade-in duration-200"
          />

          {/* Slide-In Navigation Sheet */}
          <div className="relative w-[85%] max-w-sm h-full bg-surface shadow-2xl border-r border-subtle z-50 flex flex-col justify-between p-5 animate-in slide-in-from-left duration-300 overflow-y-auto">
            {/* Drawer Header */}
            <div>
              <div className="flex items-center justify-between pb-4 mb-4 border-b border-subtle">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-emerald-700 to-emerald-500 text-white flex items-center justify-center shadow-md shadow-emerald-600/20">
                    <Leaf className="w-5 h-5" />
                  </div>
                  <div>
                    <h2 className="font-extrabold text-base text-primary-color">PlantCare</h2>
                    <p className="text-[11px] text-emerald-600 dark:text-emerald-400 font-medium">AI Plant Health Detector</p>
                  </div>
                </div>

                <button
                  onClick={() => setMobileMenuOpen(false)}
                  className="p-2 rounded-xl bg-surface-elevated text-secondary-color hover:text-primary-color border border-subtle active:scale-95 transition-all cursor-pointer"
                  aria-label="Close navigation menu"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Server Status Chip */}
              <div className="flex items-center justify-between px-3 py-2 mb-4 rounded-xl bg-surface-elevated border border-subtle text-xs">
                <div className="flex items-center gap-2">
                  <span className="relative flex h-2 w-2">
                    {serverStatus === 'online' && (
                      <>
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                      </>
                    )}
                    {serverStatus === 'waking' && (
                      <span className="animate-ping relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
                    )}
                    {serverStatus === 'checking' && (
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500 animate-pulse"></span>
                    )}
                    {serverStatus === 'offline' && (
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-zinc-400"></span>
                    )}
                  </span>
                  <span className="text-secondary-color font-medium">
                    {serverStatus === 'online' && 'AI Engine Online (Ready)'}
                    {serverStatus === 'waking' && 'Warming up Render container...'}
                    {serverStatus === 'checking' && 'Connecting to API...'}
                    {serverStatus === 'offline' && 'Offline / Resilient Mode'}
                  </span>
                </div>
              </div>

              {/* Navigation Links */}
              <div className="space-y-1">
                <div className="px-2 py-1 text-[10px] font-bold text-muted-color uppercase tracking-wider">
                  Menu
                </div>
                {navItems.map((item) => {
                  const isActive = currentPage === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => handleNavClick(item.id)}
                      className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all duration-150 cursor-pointer ${
                        isActive
                          ? 'bg-emerald-50 text-emerald-900 dark:bg-zinc-800 dark:text-emerald-300 font-semibold shadow-xs'
                          : 'text-secondary-color hover:bg-surface-elevated hover:text-primary-color'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <span className={isActive ? 'text-emerald-600 dark:text-emerald-400' : 'text-muted-color'}>
                          {item.icon}
                        </span>
                        <span>{item.label}</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        {item.badge && (
                          <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-md bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300">
                            {item.badge}
                          </span>
                        )}
                        {isActive && (
                          <span className="w-1.5 h-4 rounded-full bg-emerald-600 dark:bg-emerald-400" />
                        )}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Bottom Section: Model Selection & Theme Switcher */}
            <div className="pt-4 border-t border-subtle space-y-3 mt-4">
              {/* Model Switcher */}
              {availableModels.length > 0 && onSelectModel && (
                <div>
                  <div className="px-2 py-1 text-[10px] font-bold text-muted-color uppercase tracking-wider">
                    Active AI Model
                  </div>
                  <div className="relative mt-1">
                    <button
                      onClick={() => setShowModelMenu(!showModelMenu)}
                      className="w-full flex items-center justify-between p-2.5 rounded-xl bg-surface-elevated border border-subtle text-xs text-left"
                    >
                      <div className="flex items-center gap-2 truncate">
                        <Cpu className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
                        <span className="font-semibold text-primary-color truncate">
                          {currentModelObj?.name || 'EfficientNet-B0'}
                        </span>
                      </div>
                      <ChevronRight className={`w-3.5 h-3.5 text-muted-color transition-transform ${showModelMenu ? 'rotate-90' : ''}`} />
                    </button>

                    {showModelMenu && (
                      <div className="mt-1.5 p-1 bg-surface border border-subtle rounded-xl shadow-lg space-y-1">
                        {availableModels.map((m) => (
                          <button
                            key={m.id}
                            onClick={() => {
                              onSelectModel(m.id);
                              setShowModelMenu(false);
                            }}
                            className={`w-full flex items-center justify-between p-2 rounded-lg text-xs text-left ${
                              m.id === selectedModel
                                ? 'bg-emerald-50 text-emerald-900 dark:bg-zinc-800 dark:text-emerald-300 font-semibold'
                                : 'text-secondary-color hover:bg-surface-elevated'
                            }`}
                          >
                            <span>{m.name}</span>
                            {m.id === selectedModel && (
                              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                            )}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Quick Analyze CTA */}
              <button
                onClick={() => {
                  onNewAnalysis();
                  setMobileMenuOpen(false);
                }}
                className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-gradient-to-r from-emerald-700 to-emerald-600 hover:from-emerald-800 hover:to-emerald-700 text-white text-xs font-bold shadow-md shadow-emerald-600/20 active:scale-98 transition-all cursor-pointer"
              >
                <Plus className="w-4 h-4" />
                <span>Start New Analysis</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
