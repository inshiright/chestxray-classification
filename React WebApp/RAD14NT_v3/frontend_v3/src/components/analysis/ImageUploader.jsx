import React, { useCallback, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Image as ImageIcon, X, Scan } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function ImageUploader({ onImageSelect, uploadedImage, onClear, isAnalyzing }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragOut = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  }, []);

  const processFile = (file) => {
    if (!file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      onImageSelect({ file, preview: e.target.result });
    };
    reader.readAsDataURL(file);
  };

  const handleFileInput = (e) => {
    const file = e.target.files[0];
    if (file) processFile(file);
  };

  if (uploadedImage) {
    return (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="relative group"
      >
        <div className="relative rounded-2xl overflow-hidden border-2 border-primary/20 bg-card shadow-lg">
          <img
            src={uploadedImage.preview}
            alt="X-Ray"
            className="w-full h-auto max-h-[420px] object-contain bg-foreground/5"
          />
          <div className="absolute inset-0 bg-gradient-to-t from-foreground/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
          
          {!isAnalyzing && (
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              onClick={onClear}
              className="absolute top-3 right-3 p-2 rounded-full bg-foreground/70 text-background hover:bg-foreground/90 transition-colors backdrop-blur-sm"
            >
              <X className="w-4 h-4" />
            </motion.button>
          )}

          {isAnalyzing && (
            <div className="absolute inset-0 flex items-center justify-center bg-foreground/20 backdrop-blur-[2px]">
              <div className="relative">
                <div className="w-16 h-16 rounded-full border-4 border-primary/30 border-t-primary animate-spin" />
                <Scan className="w-6 h-6 text-primary absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2" />
              </div>
            </div>
          )}
        </div>

        <div className="mt-3 px-1">
          <p className="text-sm text-muted-foreground truncate font-inter">
            {uploadedImage.file.name}
          </p>
          <p className="text-xs text-muted-foreground/60 font-inter">
            {(uploadedImage.file.size / 1024).toFixed(1)} KB
          </p>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div
        onDragEnter={handleDragIn}
        onDragLeave={handleDragOut}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`
          relative cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-300
          min-h-[320px] flex flex-col items-center justify-center p-8
          ${isDragging
            ? "border-primary bg-primary/5 scale-[1.02]"
            : "border-border hover:border-primary/40 hover:bg-primary/[0.02]"
          }
        `}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileInput}
          className="hidden"
        />

        <AnimatePresence mode="wait">
          {isDragging ? (
            <motion.div
              key="dragging"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="flex flex-col items-center gap-4"
            >
              <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center">
                <ImageIcon className="w-8 h-8 text-primary" />
              </div>
              <p className="text-lg font-medium text-primary font-space">Drop your X-Ray here</p>
            </motion.div>
          ) : (
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-5"
            >
              <div className="relative">
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary/10 to-accent/10 flex items-center justify-center">
                  <Upload className="w-9 h-9 text-primary/70" />
                </div>
                <div className="absolute -bottom-1 -right-1 w-7 h-7 rounded-lg bg-accent/20 flex items-center justify-center">
                  <Scan className="w-4 h-4 text-accent" />
                </div>
              </div>

              <div className="text-center space-y-2">
                <p className="text-base font-medium text-foreground font-space">
                  Upload Chest X-Ray
                </p>
                <p className="text-sm text-muted-foreground font-inter">
                  Drag & drop or click to browse
                </p>
              </div>

              <div className="flex items-center gap-2 text-xs text-muted-foreground/60 font-inter">
                <span>PNG</span>
                <span className="w-1 h-1 rounded-full bg-muted-foreground/30" />
                <span>JPG</span>
                <span className="w-1 h-1 rounded-full bg-muted-foreground/30" />
                <span>DICOM</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}