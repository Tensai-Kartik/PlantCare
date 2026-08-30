import React from 'react';
import { Menu, X, Sun, Moon, Leaf, Plus, RotateCcw, Download, Printer } from 'lucide-react';
import { PageRoute, ThemeMode, AnalysisResponse } from '../../types';

interface HeaderProps {
  currentPage: PageRoute;
  onNavigate: (page: PageRoute) => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  onNewAnalysis: () => void;
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
  analysisResult,
  onExportJSON,
  onDownloadReport,
  onResetAnalysis
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = React.useState(false);

  const navItems: { id: PageRoute; label: string }[] = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'analyze', label: 'Analyze Plant' },
    { id: 'knowledge', label: 'Knowledge Base' },
    { id: 'diseases', label: 'Common Diseases' },
    { id: 'treatment', label: 'Treatment Guide' },
    { id: 'prevention', label: 'Tips & Prevention' }
  ];

  return (
    <header className="md:hidden bg-surface border-b border-subtle sticky top-0 z-40 px-4 py-3 flex items-center justify-between">
      {analysisResult && onResetAnalysis ? (
        /* Mobile Analysis Top Bar */
        <>
          <button
            onClick={onResetAnalysis}
            className="flex items-center gap-1.5 text-xs font-semibold text-secondary-color hover:text-primary-color"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Back</span>
          </button>

          <div className="flex items-center gap-1.5">
            {/* 1. Analyze Another Plant */}
            <button
              onClick={onResetAnalysis}
              className="p-2 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white shadow-xs"
              title="Analyze Another Plant"
            >
              <Plus className="w-4 h-4" />
            </button>

            {/* 2. Dark/Light switch */}
            <button
              onClick={onToggleTheme}
              className="p-2 rounded-lg bg-surface-elevated text-secondary-color border border-subtle"
              title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Mode`}
            >
              {theme === 'light' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
            </button>

            {/* 3. Export to JSON */}
            {onExportJSON && (
              <button
                onClick={onExportJSON}
                className="p-2 rounded-lg bg-surface-elevated text-secondary-color border border-subtle"
                title="Export to JSON"
              >
                <Download className="w-4 h-4" />
              </button>
            )}

            {/* 4. Download Report */}
            {onDownloadReport && (
              <button
                onClick={onDownloadReport}
                className="p-2 rounded-lg bg-surface-elevated text-primary-color border border-subtle"
                title="Download Report"
              >
                <Printer className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
              </button>
            )}
          </div>
        </>
      ) : (
        /* Standard Mobile Brand + Actions Header */
        <>
          <div 
            onClick={() => onNavigate('dashboard')}
            className="flex items-center gap-2.5 cursor-pointer"
          >
            <div className="w-8 h-8 rounded-lg bg-emerald-600 dark:bg-emerald-500 text-white flex items-center justify-center">
              <Leaf className="w-5 h-5" />
            </div>
            <div>
              <span className="font-bold text-base text-primary-color block leading-none">PlantCare</span>
              <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-medium">AI Plant Health</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={onNewAnalysis}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold shadow-xs"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Analyze</span>
            </button>

            <button
              onClick={onToggleTheme}
              className="p-2 rounded-lg bg-surface-elevated text-secondary-color border border-subtle"
            >
              {theme === 'light' ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
            </button>

            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg bg-surface-elevated text-secondary-color border border-subtle"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </>
      )}

      {/* Mobile Dropdown Menu */}
      {mobileMenuOpen && (
        <div className="absolute top-full left-0 w-full bg-surface border-b border-subtle shadow-xl p-4 space-y-2 z-50 animate-in fade-in slide-in-from-top-2">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => {
                onNavigate(item.id);
                setMobileMenuOpen(false);
              }}
              className={`w-full text-left px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                currentPage === item.id
                  ? 'bg-emerald-50 text-emerald-800 dark:bg-zinc-800 dark:text-emerald-400 font-semibold'
                  : 'text-secondary-color hover:bg-surface-elevated'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      )}
    </header>
  );
};
