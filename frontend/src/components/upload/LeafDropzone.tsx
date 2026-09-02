import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, Camera, AlertCircle, RefreshCw, ArrowRight, Car } from 'lucide-react';
import { CameraModal } from '../camera/CameraModal';

interface LeafDropzoneProps {
  onImageSelected: (file: File) => void;
  isLoading?: boolean;
}

export const LeafDropzone: React.FC<LeafDropzoneProps> = ({ onImageSelected, isLoading = false }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isCameraOpen, setIsCameraOpen] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  const validateAndHandleFile = (file: File) => {
    setErrorMsg(null);
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
    if (!validTypes.includes(file.type)) {
      setErrorMsg('Unsupported format. Please upload a JPG, PNG, or WEBP image.');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setErrorMsg('File size exceeds 10 MB. Please upload a smaller image.');
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndHandleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndHandleFile(e.target.files[0]);
    }
  };

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setErrorMsg(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (cameraInputRef.current) cameraInputRef.current.value = '';
  };

  const handleStartAnalysis = () => {
    if (selectedFile) {
      onImageSelected(selectedFile);
    }
  };

  const handleTestCarDemo = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      const resp = await fetch('/real_car.jpg');
      const blob = await resp.blob();
      const file = new File([blob], 'sports_car_specimen.jpg', { type: 'image/jpeg' });
      validateAndHandleFile(file);
    } catch (err) {
      console.error('Failed to load sample car:', err);
    }
  };

  return (
    <div className="w-full space-y-4">
      {/* Hidden file inputs */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={handleFileInput}
        className="hidden"
      />
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleFileInput}
        className="hidden"
      />

      {/* Main Drag & Drop Zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !previewUrl && fileInputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-3xl p-6 sm:p-10 flex flex-col items-center justify-center text-center transition-all duration-200 cursor-pointer overflow-hidden ${
          isDragOver
            ? 'border-emerald-500 bg-emerald-50/50 dark:bg-zinc-900/60 scale-[1.01]'
            : previewUrl
            ? 'border-emerald-500/50 bg-surface custom-shadow'
            : 'border-strong hover:border-emerald-500/80 bg-surface-subtle hover:bg-surface'
        }`}
      >
        {previewUrl && selectedFile ? (
          /* Preview State */
          <div className="w-full flex flex-col items-center">
            <div className="relative group max-w-sm rounded-2xl overflow-hidden shadow-lg border border-subtle">
              <img
                src={previewUrl}
                alt="Selected specimen"
                className="w-full h-64 object-cover"
              />
              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-3">
                <button
                  type="button"
                  onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
                  className="px-3 py-1.5 rounded-lg bg-white/90 text-black text-xs font-semibold hover:bg-white shadow-md cursor-pointer"
                >
                  Change
                </button>
                <button
                  type="button"
                  onClick={handleClear}
                  className="px-3 py-1.5 rounded-lg bg-red-600 text-white text-xs font-semibold hover:bg-red-700 shadow-md cursor-pointer"
                >
                  Remove
                </button>
              </div>
            </div>

            <div className="mt-4 flex flex-wrap items-center justify-center gap-2 text-xs text-muted-color">
              <span className="font-semibold text-primary-color">{selectedFile.name}</span>
              <span>•</span>
              <span>{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</span>
            </div>

            <div className="mt-6 flex gap-3">
              <button
                type="button"
                onClick={handleClear}
                className="px-4 py-2.5 rounded-xl border border-subtle hover:bg-surface-elevated text-secondary-color text-sm font-medium transition-all cursor-pointer"
              >
                Clear
              </button>
              <button
                type="button"
                onClick={handleStartAnalysis}
                disabled={isLoading}
                className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-semibold text-sm shadow-md shadow-emerald-600/20 hover:scale-[1.02] active:scale-[0.98] transition-all cursor-pointer"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <span>Assess & Diagnose</span>
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </div>
          </div>
        ) : (
          /* Empty / Upload Prompt State */
          <div className="flex flex-col items-center max-w-md">
            <div className="w-16 h-16 rounded-2xl bg-emerald-100 dark:bg-zinc-800 text-emerald-600 dark:text-emerald-400 flex items-center justify-center mb-4 shadow-inner">
              <UploadCloud className="w-8 h-8" />
            </div>

            <h3 className="text-lg font-bold text-primary-color mb-1">
              Drag & drop a leaf image here
            </h3>
            <p className="text-sm text-secondary-color mb-4">
              or <span className="text-emerald-600 dark:text-emerald-400 font-semibold underline decoration-emerald-500/40">click to browse</span> from your device
            </p>

            <div className="flex items-center gap-2 mb-4">
              <span className="text-[11px] font-medium tracking-wide uppercase px-2 py-0.5 rounded-md bg-surface-elevated border border-subtle text-muted-color">
                JPG • PNG • WEBP (Max 10MB)
              </span>
            </div>

            {/* Quick Action Buttons */}
            <div className="flex flex-wrap items-center justify-center gap-2">
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface border border-strong hover:border-emerald-500 text-xs font-semibold text-primary-color shadow-xs transition-all cursor-pointer"
              >
                <ImageIcon className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                <span>Choose File</span>
              </button>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); setIsCameraOpen(true); }}
                className="flex items-center gap-2 px-4 py-2 rounded-xl bg-surface border border-strong hover:border-emerald-500 text-xs font-semibold text-primary-color shadow-xs transition-all cursor-pointer"
              >
                <Camera className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                <span>Take Photo</span>
              </button>
              <button
                type="button"
                onClick={handleTestCarDemo}
                className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-surface-elevated border border-subtle hover:border-red-400 text-xs font-medium text-secondary-color hover:text-red-500 shadow-xs transition-all cursor-pointer"
                title="Test how PlantCare detects and rejects non-plant images (e.g. cars)"
              >
                <Car className="w-3.5 h-3.5 text-red-500" />
                <span>Try Car Demo</span>
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Live Device Camera Modal */}
      <CameraModal
        isOpen={isCameraOpen}
        onClose={() => setIsCameraOpen(false)}
        onCapture={(file) => {
          validateAndHandleFile(file);
          setIsCameraOpen(false);
        }}
      />

      {/* Error Message */}
      {errorMsg && (
        <div className="flex items-center gap-2 p-3 rounded-xl bg-red-50 dark:bg-zinc-900 text-red-700 dark:text-red-300 text-xs border border-red-200 dark:border-zinc-800">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Tip Banner */}
      <div className="flex items-center gap-2.5 px-4 py-2.5 rounded-xl bg-emerald-50/60 dark:bg-zinc-900 border border-emerald-200/60 dark:border-zinc-800 text-emerald-900 dark:text-zinc-200 text-xs">
        <span className="font-bold text-emerald-700 dark:text-emerald-400 uppercase text-[10px] tracking-wider shrink-0 px-1.5 py-0.5 rounded bg-emerald-100 dark:bg-zinc-800">
          Guardrail Active
        </span>
        <p className="leading-snug">
          PlantCare AI automatically verifies plant foliage presence to prevent false disease misdiagnoses on non-plant photos.
        </p>
      </div>
    </div>
  );
};
