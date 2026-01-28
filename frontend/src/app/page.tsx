"use client";

import React, { useEffect, useState, useRef, useCallback } from "react";
import { ShieldAlert, CheckCircle, Activity, Camera, AlertTriangle, Video, VideoOff, BarChart3, Image as ImageIcon, Download } from "lucide-react";

export default function Dashboard() {
  const [stream, setStream] = useState<string | null>(null);
  const [alerts, setAlerts] = useState<any[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [latency, setLatency] = useState(0);
  const [fps, setFps] = useState(0);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [stats, setStats] = useState<any>(null);
  const [violationFiles, setViolationFiles] = useState<string[]>([]);
  const [isEmergencyMode, setIsEmergencyMode] = useState(false);
  const [criticalEvents, setCriticalEvents] = useState<any[]>([]);

  const ws = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isStreamingRef = useRef(false);
  const frameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(Date.now());
  const pendingFrameRef = useRef(false);

  // Voice Announcement Helper
  const speak = (text: string) => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1;
      window.speechSynthesis.speak(utterance);
    }
  };

  // Connect to WebSocket
  useEffect(() => {
    const backendHost = "127.0.0.1:8000";
    ws.current = new WebSocket(`ws://${backendHost}/ws`);

    ws.current.onopen = () => setIsConnected(true);
    ws.current.onclose = () => {
      setIsConnected(false);
      stopWebcam();
    };

    ws.current.onmessage = (event) => {
      console.log("📥 Received response from backend");
      pendingFrameRef.current = false;

      const data = JSON.parse(event.data);
      if (data.error) {
        console.error("Backend error:", data.error, data.details);
        return;
      }
      setStream(data.annotated_frame);

      // Emergency Mode Logic
      if (data.response_actions?.emergency_mode) {
        setIsEmergencyMode(true);
      } else {
        setIsEmergencyMode(false);
      }

      // Voice Announcement
      if (data.response_actions?.voice_announcement) {
        speak(data.response_actions.voice_announcement);
      }

      // Update Alerts
      if (data.critical_events?.length > 0) {
        setCriticalEvents(data.critical_events);
      } else {
        setCriticalEvents([]);
      }

      // Calculate FPS
      frameCountRef.current++;
      const now = Date.now();
      if (now - lastFpsTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastFpsTimeRef.current = now;
      }

      if (data.alert && data.violations?.length > 0) {
        const newAlert = {
          id: Date.now(),
          time: new Date().toLocaleTimeString(),
          violations: data.violations,
        };
        setAlerts((prev) => [newAlert, ...prev].slice(0, 5));
      }
    };

    return () => {
      ws.current?.close();
      stopWebcam();
    };
  }, []);

  // Fetch Analytics
  const fetchAnalytics = async () => {
    try {
      const backendUrl = "http://127.0.0.1:8000";
      const res = await fetch(`${backendUrl}/api/analytics`);
      const data = await res.json();
      setStats(data);

      const vRes = await fetch("http://127.0.0.1:8000/api/violations");
      const vData = await vRes.json();
      setViolationFiles(vData.violations);
    } catch (err) {
      console.error("Failed to fetch analytics:", err);
    }
  };

  useEffect(() => {
    if (showAnalytics) {
      fetchAnalytics();
      const interval = setInterval(fetchAnalytics, 5000);
      return () => clearInterval(interval);
    }
  }, [showAnalytics]);

  // Start webcam capture
  const startWebcam = useCallback(async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 },
          facingMode: "environment"
        }
      });

      console.log("📸 Camera stream acquired");
      streamRef.current = mediaStream;
      isStreamingRef.current = true;
      setIsStreaming(true);

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        await videoRef.current.play();
      }

      // Start optimized frame loop
      const sendFrame = () => {
        // CRITICAL: Check the ref, not state (immediate check)
        if (!isStreamingRef.current) {
          console.log("Stream stopped, exiting loop");
          return;
        }

        if (!videoRef.current || !canvasRef.current || !ws.current) {
          requestAnimationFrame(sendFrame);
          return;
        }

        if (ws.current.readyState !== WebSocket.OPEN) {
          requestAnimationFrame(sendFrame);
          return;
        }

        // Skip frame if previous one hasn't been processed yet (prevents lag)
        if (pendingFrameRef.current) {
          requestAnimationFrame(sendFrame);
          return;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        if (ctx && video.readyState === video.HAVE_ENOUGH_DATA) {
          canvas.width = 480;
          canvas.height = 480;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          const startTime = performance.now();
          // Lower quality for faster encoding - reduced to 0.5
          const frameData = canvas.toDataURL("image/jpeg", 0.5);

          if (frameCountRef.current % 30 === 0) {
            console.log(`📤 Sending frame ${frameCountRef.current}, data length: ${frameData.length}`);
          }

          pendingFrameRef.current = true; // Mark frame as pending

          // Safety timeout to reset pending frame if backend doesn't respond
          setTimeout(() => {
            if (pendingFrameRef.current) {
              console.warn("⏱️ Frame response timeout, resetting...");
              pendingFrameRef.current = false;
            }
          }, 2000);

          ws.current.send(frameData);

          setLatency(Math.round(performance.now() - startTime));
        }

        // Continue loop only if still streaming
        if (isStreamingRef.current) {
          requestAnimationFrame(sendFrame);
        }
      };

      // Start after video is ready
      setTimeout(sendFrame, 300);

    } catch (err) {
      console.error("Webcam error:", err);
      alert("Could not access webcam. Please check permissions.");
      isStreamingRef.current = false;
      setIsStreaming(false);
    }
  }, []);

  // Stop webcam - FIXED: properly stops the loop
  const stopWebcam = useCallback(() => {
    console.log("Stopping webcam...");

    // CRITICAL: Set ref first to stop the loop immediately
    isStreamingRef.current = false;
    setIsStreaming(false);

    // Stop all media tracks
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.stop();
        console.log("Track stopped:", track.kind);
      });
      streamRef.current = null;
    }

    // Clear video source
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // Clear displayed frame
    setStream(null);
    pendingFrameRef.current = false;

    console.log("Webcam stopped");
  }, []);

  return (
    <div className={`min-h-screen transition-colors duration-500 p-8 font-sans ${isEmergencyMode ? 'bg-red-950 animate-pulse-slow' : 'bg-slate-950'} text-slate-100`}>
      {/* Hidden video and canvas for capture */}
      <video ref={videoRef} className="hidden" playsInline muted />
      <canvas ref={canvasRef} className="hidden" />

      {/* Header */}
      <div className={`flex justify-between items-center mb-10 border-b pb-6 ${isEmergencyMode ? 'border-red-500/50' : 'border-slate-800'}`}>
        <div>
          <h1 className={`text-4xl font-black ${isEmergencyMode ? 'text-red-500' : 'bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent'}`}>
            GUARDIAN VISION {isEmergencyMode && "• EMERGENCY MODE"}
          </h1>
          <p className="text-slate-400 text-sm mt-1 uppercase tracking-widest">
            Visual Compliance Auditor • Real-Time Engine
          </p>
        </div>
        <div className="flex items-center gap-4">
          {/* Webcam Toggle Button */}
          <button
            onClick={isStreaming ? stopWebcam : startWebcam}
            disabled={!isConnected}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl font-bold text-sm transition-all ${isStreaming
              ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30 hover:bg-rose-500/30'
              : 'bg-blue-500/20 text-blue-400 border border-blue-500/30 hover:bg-blue-500/30'
              } ${!isConnected ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {isStreaming ? <VideoOff size={16} /> : <Video size={16} />}
            {isStreaming ? "STOP CAMERA" : "START CAMERA"}
          </button>

          <button
            onClick={() => setShowAnalytics(!showAnalytics)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl font-bold text-sm transition-all ${showAnalytics
              ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
              : 'bg-slate-800 text-slate-400 border border-slate-700 hover:bg-slate-700'
              }`}
          >
            <BarChart3 size={16} />
            {showAnalytics ? "VIEW LIVE FEED" : "ANALYTICS"}
          </button>

          <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold ${isConnected ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'}`}>
            <Activity size={14} className={isConnected ? "animate-pulse" : ""} />
            {isConnected ? "SYSTEM ONLINE" : "ENGINE OFFLINE"}
          </div>
        </div>
      </div>

      {/* Toggleable Views */}
      <div className="lg:col-span-3">
        {!showAnalytics ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Main Feed */}
            <div className="lg:col-span-2 space-y-6">
              <div className="relative aspect-video bg-slate-900 rounded-3xl overflow-hidden border border-slate-800 shadow-2xl flex items-center justify-center">
                {stream ? (
                  <img src={stream} className="w-full h-full object-cover" alt="Real-time Feed" />
                ) : (
                  <div className="flex flex-col items-center gap-4 text-slate-600">
                    <Camera size={48} />
                    <p>{isConnected ? "Click 'Start Camera' to begin" : "Connecting to backend..."}</p>
                  </div>
                )}
                <div className="absolute top-4 left-4 flex gap-2">
                  <span className="bg-black/60 backdrop-blur-md px-3 py-1 rounded-lg text-[10px] font-bold border border-white/10 uppercase italic">
                    {isStreaming ? "🔴 LIVE" : "STANDBY"}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <StatsCard label="Frame Rate" value={`${fps} FPS`} color="text-blue-400" />
                <StatsCard label="Encode Latency" value={`${latency}ms`} color="text-emerald-400" />
                <StatsCard label="Detections" value={alerts.length > 0 ? `${alerts.length} alerts` : "0"} color="text-slate-400" />
              </div>
            </div>

            {/* Sidebar Alerts */}
            <div className="space-y-6">
              {/* Critical Events Section */}
              {criticalEvents.length > 0 && (
                <div className="bg-red-900/40 rounded-3xl p-6 border-2 border-red-500 animate-pulse">
                  <div className="flex items-center gap-3 mb-4">
                    <ShieldAlert className="text-white" />
                    <h2 className="text-xl font-bold text-white uppercase italic">CRITICAL THREATS</h2>
                  </div>
                  <div className="space-y-3">
                    {criticalEvents.map((event, i) => (
                      <div key={i} className="bg-white/10 p-3 rounded-xl border border-white/20">
                        <div className="font-black text-white">{event.type}</div>
                        <div className="text-red-200 text-[10px] font-bold uppercase">{event.location}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="bg-slate-900/50 rounded-3xl p-6 border border-slate-800 backdrop-blur-sm min-h-[300px]">
                <div className="flex items-center gap-3 mb-6">
                  <ShieldAlert className="text-rose-500" />
                  <h2 className="text-xl font-bold italic uppercase tracking-tight">Active Violations</h2>
                </div>

                <div className="space-y-4">
                  {alerts.length === 0 ? (
                    <div className="bg-emerald-500/5 border border-emerald-500/10 p-4 rounded-xl flex items-center gap-3 text-emerald-500">
                      <CheckCircle size={20} />
                      <span className="text-sm font-medium">No active PPE violations detected.</span>
                    </div>
                  ) : (
                    alerts.map((alert) => (
                      <div key={alert.id} className="bg-rose-500/5 border border-rose-500/20 p-4 rounded-2xl animate-in slide-in-from-right duration-300">
                        <div className="flex justify-between items-start mb-2">
                          <div className="flex items-center gap-2 text-rose-500">
                            <AlertTriangle size={16} />
                            <span className="font-bold text-xs">CRITICAL ALERT</span>
                          </div>
                          <span className="text-[10px] text-slate-500 font-mono">{alert.time}</span>
                        </div>
                        <div className="text-red-100 text-sm font-semibold mb-1">PPE Compliance Failure</div>
                        <div className="flex gap-2 mt-2 flex-wrap">
                          {alert.violations?.[0]?.violations?.map((v: string) => (
                            <span key={v} className="bg-rose-500/20 text-rose-400 px-2 py-0.5 rounded text-[10px] font-bold uppercase">
                              Missing {v}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Analytics View */
          <div className="animate-in fade-in duration-500">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
              <StatsCard label="Total Violations" value={stats?.total_violations || "0"} color="text-rose-400" />
              <StatsCard label="Hardhat Violations" value={stats?.violations_by_type?.Hardhat || "0"} color="text-amber-400" />
              <StatsCard label="Vest Violations" value={stats?.violations_by_type?.["Safety Vest"] || "0"} color="text-blue-400" />
              <StatsCard label="Compliance Score" value="84%" color="text-emerald-400" />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Evidence Gallery */}
              <div className="lg:col-span-2 bg-slate-900/50 rounded-3xl p-8 border border-slate-800">
                <div className="flex justify-between items-center mb-6">
                  <div className="flex items-center gap-3">
                    <ImageIcon className="text-blue-400" />
                    <h2 className="text-xl font-bold uppercase italic tracking-tight">Evidence Gallery</h2>
                  </div>
                  <button className="flex items-center gap-2 text-xs font-bold text-slate-400 hover:text-white transition-colors">
                    <Download size={14} /> EXPORT ALL (CSV)
                  </button>
                </div>

                {violationFiles.length === 0 ? (
                  <div className="aspect-video flex flex-col items-center justify-center text-slate-600 border-2 border-dashed border-slate-800 rounded-2xl">
                    <ImageIcon size={48} className="mb-4 opacity-20" />
                    <p>No violation evidence captured yet.</p>
                  </div>
                ) : (
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    {violationFiles.map((file) => (
                      <div key={file} className="group relative aspect-square bg-slate-800 rounded-xl overflow-hidden border border-slate-700 hover:border-blue-500/50 transition-all">
                        <img
                          src={`http://127.0.0.1:8000/violations/${file}`}
                          className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
                          alt="Violation"
                        />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity p-3 flex flex-col justify-end">
                          <p className="text-[8px] font-mono text-slate-300 truncate">{file}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Daily Distribution */}
              <div className="bg-slate-900/50 rounded-3xl p-8 border border-slate-800">
                <div className="flex items-center gap-3 mb-6">
                  <Activity className="text-emerald-400" />
                  <h2 className="text-xl font-bold uppercase italic tracking-tight">Compliance Feed</h2>
                </div>
                <div className="space-y-4">
                  {Object.entries(stats?.daily_trends || {}).map(([day, data]: [string, any]) => (
                    <div key={day} className="flex justify-between items-center p-3 rounded-xl bg-slate-800/50 border border-slate-700/50">
                      <div>
                        <p className="text-[10px] font-bold text-slate-500">{day}</p>
                        <p className="text-xs font-semibold">{data.person_frames} instances</p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs font-bold text-rose-400">{data.violations} violations</p>
                        <div className="w-20 h-1 bg-slate-700 rounded-full mt-1 overflow-hidden">
                          <div
                            className="h-full bg-rose-500"
                            style={{ width: `${Math.min(100, (data.violations / (data.person_frames || 1)) * 1000)}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatsCard({ label, value, color }: { label: string, value: string, color: string }) {
  return (
    <div className="bg-slate-900/50 p-4 rounded-2xl border border-slate-800">
      <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1">{label}</div>
      <div className={`text-xl font-black ${color}`}>{value}</div>
    </div>
  );
}
