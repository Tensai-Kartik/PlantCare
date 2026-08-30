import React from 'react';
import { 
  LayoutDashboard, 
  ScanLine, 
  BookOpen, 
  ShieldAlert, 
  Stethoscope, 
  Sprout, 
  Cpu, 
  Leaf,
  CheckCircle2
} from 'lucide-react';
import { PageRoute, ThemeMode, ModelMetadata } from '../../types';

interface SidebarProps {
  currentPage: PageRoute;
  onNavigate: (page: PageRoute) => void;
  theme: ThemeMode;
  onToggleTheme: () => void;
  selectedModel: string;
  availableModels: ModelMetadata[];
  onSelectModel: (modelId: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentPage,
  onNavigate,
  theme,
  onToggleTheme,
  selectedModel,
  availableModels,
  onSelectModel
}) => {
  const [showModelMenu, setShowModelMenu] = React.useState(false);

  const navItems: { id: PageRoute; label: string; icon: React.ReactNode; badge?: string }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard className="w-5 h-5" /> },
    { id: 'analyze', label: 'Analyze Plant', icon: <ScanLine className="w-5 h-5" /> },
    { id: 'knowledge', label: 'Knowledge Base', icon: <BookOpen className="w-5 h-5" /> },
    { id: 'diseases', label: 'Common Diseases', icon: <ShieldAlert className="w-5 h-5" /> },
    { id: 'treatment', label: 'Treatment Guide', icon: <Stethoscope className="w-5 h-5" /> },
    { id: 'prevention', label: 'Tips & Prevention', icon: <Sprout className="w-5 h-5" /> }
  ];

  const currentModelObj = availableModels.find(m => m.id === selectedModel) || availableModels[0];

  return (
    <aside className="w-64 h-screen bg-surface border-r border-subtle flex flex-col justify-between p-4 select-none shrink-0 transition-colors duration-200">
      {/* Brand Header */}
      <div>
        <div 
          onClick={() => onNavigate('dashboard')}
          className="flex items-center gap-3 px-2 py-3 mb-6 cursor-pointer group"
        >
          <div className="w-10 h-10 rounded-xl bg-emerald-600 dark:bg-emerald-500 text-white flex items-center justify-center shadow-md shadow-emerald-500/20 group-hover:scale-105 transition-transform">
            <Leaf className="w-6 h-6" />
          </div>
          <div>
            <h1 className="font-bold text-lg leading-tight tracking-tight text-primary-color flex items-center gap-1.5">
              PlantCare
            </h1>
            <p className="text-xs font-medium text-emerald-600 dark:text-emerald-400">
              AI Plant Health Detector
            </p>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="space-y-1">
          {navItems.map((item) => {
            const isActive = currentPage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-emerald-50 text-emerald-800 dark:bg-zinc-800/90 dark:text-emerald-400 font-semibold shadow-xs'
                    : 'text-secondary-color hover:bg-surface-elevated hover:text-primary-color'
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className={isActive ? 'text-emerald-600 dark:text-emerald-400' : 'text-muted-color'}>
                    {item.icon}
                  </span>
                  <span>{item.label}</span>
                </div>
                {isActive && (
                  <span className="w-1.5 h-4 rounded-full bg-emerald-600 dark:bg-emerald-400" />
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer Area: Model Selector */}
      <div className="pt-4 border-t border-subtle">
        {/* Model Selector Card */}
        <div className="relative">
          <button
            onClick={() => setShowModelMenu(!showModelMenu)}
            className="w-full flex items-center justify-between p-2.5 rounded-xl bg-surface-elevated hover:bg-surface-subtle border border-subtle text-left transition-all text-xs"
          >
            <div className="flex items-center gap-2 truncate">
              <div className="p-1 rounded-md bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                <Cpu className="w-4 h-4" />
              </div>
              <div className="truncate">
                <p className="font-semibold text-primary-color truncate">
                  {currentModelObj?.name || 'EfficientNet-B0'}
                </p>
                <p className="text-[10px] text-muted-color">
                  {currentModelObj?.accuracy ? `${currentModelObj.accuracy}% test acc` : 'Active CV Model'}
                </p>
              </div>
            </div>
            <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 dark:bg-zinc-800 dark:text-emerald-400">
              AI
            </span>
          </button>

          {/* Model Selection Dropdown */}
          {showModelMenu && (
            <div className="absolute bottom-full left-0 w-full mb-2 p-1.5 bg-surface border border-subtle rounded-xl shadow-xl z-50 space-y-1">
              <div className="px-2 py-1 text-[11px] font-bold text-muted-color uppercase tracking-wider">
                Select AI Model
              </div>
              {availableModels.map((m) => (
                <button
                  key={m.id}
                  onClick={() => {
                    onSelectModel(m.id);
                    setShowModelMenu(false);
                  }}
                  className={`w-full flex items-center justify-between p-2 rounded-lg text-xs text-left transition-colors ${
                    m.id === selectedModel
                      ? 'bg-emerald-50 text-emerald-800 dark:bg-zinc-800 dark:text-emerald-400 font-semibold'
                      : 'text-secondary-color hover:bg-surface-elevated'
                  }`}
                >
                  <div>
                    <p className="font-medium text-primary-color">{m.name}</p>
                    <p className="text-[10px] text-muted-color">
                      {m.accuracy ? `${m.accuracy}% accuracy • ` : ''} {m.latency_ms ? `${m.latency_ms}ms CPU` : ''}
                    </p>
                  </div>
                  {m.id === selectedModel && (
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};
