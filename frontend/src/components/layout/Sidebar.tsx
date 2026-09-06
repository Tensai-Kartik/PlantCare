import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  ScanLine, 
  BookOpen, 
  ShieldAlert, 
  Stethoscope, 
  Sprout, 
  Cpu, 
  Leaf, 
  CheckCircle2, 
  ChevronDown
} from 'lucide-react';
import { PageRoute, ModelMetadata } from '../../types';
import { subscribeServerStatus, ServerWarmupStatus } from '../../services/api';

interface SidebarProps {
  currentPage: PageRoute;
  onNavigate: (page: PageRoute) => void;
  selectedModel: string;
  availableModels: ModelMetadata[];
  onSelectModel: (modelId: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentPage,
  onNavigate,
  selectedModel,
  availableModels,
  onSelectModel
}) => {
  const [showModelMenu, setShowModelMenu] = useState(false);
  const [serverStatus, setServerStatus] = useState<ServerWarmupStatus>('checking');

  useEffect(() => {
    const unsubscribe = subscribeServerStatus(setServerStatus);
    return () => unsubscribe();
  }, []);

  const navItems: { id: PageRoute; label: string; icon: React.ReactNode; badge?: string }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard className="w-5 h-5" /> },
    { id: 'analyze', label: 'Analyze Plant', icon: <ScanLine className="w-5 h-5" />, badge: 'AI' },
    { id: 'knowledge', label: 'Knowledge Base', icon: <BookOpen className="w-5 h-5" /> },
    { id: 'diseases', label: 'Common Diseases', icon: <ShieldAlert className="w-5 h-5" /> },
    { id: 'treatment', label: 'Treatment Guide', icon: <Stethoscope className="w-5 h-5" /> },
    { id: 'prevention', label: 'Tips & Prevention', icon: <Sprout className="w-5 h-5" /> }
  ];

  const currentModelObj = availableModels.find(m => m.id === selectedModel) || availableModels[0];

  return (
    <aside className="w-64 lg:w-72 h-screen sticky top-0 bg-surface border-r border-subtle flex flex-col justify-between p-4 lg:p-5 select-none shrink-0 transition-colors duration-200 overflow-y-auto z-30">
      {/* Top Brand & Navigation Links */}
      <div>
        {/* Brand Header */}
        <div 
          onClick={() => onNavigate('dashboard')}
          className="flex items-center gap-3 px-2 py-3 mb-6 cursor-pointer group rounded-2xl hover:bg-surface-elevated/60 transition-colors"
        >
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-700 to-emerald-500 text-white flex items-center justify-center shadow-md shadow-emerald-600/25 group-hover:scale-105 transition-transform">
            <Leaf className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <h1 className="font-extrabold text-lg leading-tight tracking-tight text-primary-color">
                PlantCare
              </h1>
              <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.2 rounded bg-emerald-100 text-emerald-800 dark:bg-emerald-950/80 dark:text-emerald-300">
                v1.2
              </span>
            </div>
            <p className="text-[11px] font-medium text-emerald-600 dark:text-emerald-400 mt-0.5">
              AI Plant Health & Pathology
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
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all duration-150 cursor-pointer ${
                  isActive
                    ? 'bg-emerald-50 text-emerald-900 dark:bg-zinc-800/90 dark:text-emerald-300 font-semibold shadow-xs'
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
        </nav>
      </div>

      {/* Footer Area: Live Server Connection Status & Model Selector (No dark/light toggle) */}
      <div className="pt-4 border-t border-subtle space-y-2.5">
        {/* Live Server Status Pill */}
        <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-surface-elevated border border-subtle text-[11px]">
          <div className="flex items-center gap-2.5">
            <span className="relative flex h-2 w-2 shrink-0">
              {serverStatus === 'online' && (
                <>
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </>
              )}
              {serverStatus === 'waking' && (
                <>
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
                </>
              )}
              {serverStatus === 'checking' && (
                <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500 animate-pulse"></span>
              )}
              {serverStatus === 'offline' && (
                <span className="relative inline-flex rounded-full h-2 w-2 bg-zinc-400"></span>
              )}
            </span>
            <span className="text-secondary-color font-medium truncate">
              {serverStatus === 'online' && 'AI Engine Live'}
              {serverStatus === 'waking' && 'Warming up server...'}
              {serverStatus === 'checking' && 'Connecting to API...'}
              {serverStatus === 'offline' && 'Resilient Offline Mode'}
            </span>
          </div>
          <span className="text-[10px] text-muted-color shrink-0 font-medium">
            {serverStatus === 'online' ? 'Ready' : serverStatus === 'waking' ? 'Render' : ''}
          </span>
        </div>

        {/* Model Selector Card */}
        <div className="relative">
          <button
            onClick={() => setShowModelMenu(!showModelMenu)}
            className="w-full flex items-center justify-between p-2.5 rounded-xl bg-surface-elevated hover:bg-surface-subtle border border-subtle text-left transition-all text-xs cursor-pointer group"
          >
            <div className="flex items-center gap-2 truncate">
              <div className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 shrink-0">
                <Cpu className="w-4 h-4" />
              </div>
              <div className="truncate">
                <p className="font-semibold text-primary-color truncate">
                  {currentModelObj?.name || 'EfficientNet-B0'}
                </p>
                <p className="text-[10px] text-muted-color truncate">
                  {currentModelObj?.accuracy ? `${currentModelObj.accuracy}% accuracy` : 'Active CV Model'}
                </p>
              </div>
            </div>
            <ChevronDown className={`w-3.5 h-3.5 text-muted-color transition-transform ${showModelMenu ? 'rotate-180' : ''}`} />
          </button>

          {/* Model Selection Dropdown */}
          {showModelMenu && (
            <div className="absolute bottom-full left-0 w-full mb-2 p-1.5 bg-surface border border-subtle rounded-2xl shadow-xl z-50 space-y-1 animate-in fade-in zoom-in-95 duration-150">
              <div className="px-2.5 py-1 text-[11px] font-bold text-muted-color uppercase tracking-wider">
                Select Active Model
              </div>
              {availableModels.map((m) => (
                <button
                  key={m.id}
                  onClick={() => {
                    onSelectModel(m.id);
                    setShowModelMenu(false);
                  }}
                  className={`w-full flex items-center justify-between p-2 rounded-xl text-xs text-left transition-colors cursor-pointer ${
                    m.id === selectedModel
                      ? 'bg-emerald-50 text-emerald-900 dark:bg-zinc-800 dark:text-emerald-300 font-semibold'
                      : 'text-secondary-color hover:bg-surface-elevated'
                  }`}
                >
                  <div className="truncate pr-2">
                    <p className="font-medium text-primary-color truncate">{m.name}</p>
                    <p className="text-[10px] text-muted-color">
                      {m.accuracy ? `${m.accuracy}% acc` : ''} • {m.latency_ms ? `${m.latency_ms}ms` : ''}
                    </p>
                  </div>
                  {m.id === selectedModel && (
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
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
