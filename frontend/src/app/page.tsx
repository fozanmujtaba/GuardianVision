"use client";

/* eslint-disable @next/next/no-img-element -- live blob frames and evidence URLs should not be optimized */
import React, { useEffect, useState, useRef, useCallback } from "react";
import {
  ShieldAlert, CheckCircle, Activity, Camera, AlertTriangle,
  Video, VideoOff, BarChart3, Image as ImageIcon, Download,
  Users, Clock, Zap, Shield, Radio, Eye, Cpu,
} from "lucide-react";

// Per-class badge colours for the detection tag strip
const CLASS_COLOR: Record<string, string> = {
  "Person":           "bg-cyan-500/15 text-cyan-400 border-cyan-500/30",
  "Hardhat":          "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  "Safety Vest":      "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  "Mask":             "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  "NO-Hardhat":       "bg-rose-500/15 text-rose-400 border-rose-500/30",
  "NO-Safety Vest":   "bg-rose-500/15 text-rose-400 border-rose-500/30",
  "NO-Mask":          "bg-rose-500/15 text-rose-400 border-rose-500/30",
  "Fire":             "bg-red-500/15 text-red-400 border-red-500/30",
  "Smoke":            "bg-orange-500/15 text-orange-400 border-orange-500/30",
  "Fall Detected":    "bg-red-500/15 text-red-400 border-red-500/30",
  "Safety Cone":      "bg-amber-500/15 text-amber-400 border-amber-500/30",
  "vehicle":          "bg-blue-500/15 text-blue-400 border-blue-500/30",
  "machinery":        "bg-purple-500/15 text-purple-400 border-purple-500/30",
};

type Detection = {
  id?: number | string;
  bbox: number[];
  conf: number;
  class: number;
  class_name: string;
};

type Violation = {
  person_id: number | string;
  bbox: number[];
  violations: string[];
};

type CriticalEvent = {
  type: string;
  location: string;
  bbox?: number[];
};

type ResponseActions = {
  voice_announcement?: string | false;
  emergency_mode?: boolean;
  notifications_sent?: boolean;
};

type StreamMetadata = {
  detections?: Detection[];
  violations?: Violation[];
  critical_events?: CriticalEvent[];
  alert?: boolean;
  response_actions?: ResponseActions | null;
  device?: string;
};

type DailyTrend = {
  person_frames?: number;
  violations?: number;
};

type AnalyticsStats = {
  total_violations?: number;
  violations_by_type?: Record<string, number>;
  daily_trends?: Record<string, DailyTrend>;
};

type AlertItem = {
  id: number;
  time: string;
  violations: Violation[];
};

type ViolationsResponse = {
  violations?: string[];
};

function formatUptime(s: number) {
  const h = String(Math.floor(s / 3600)).padStart(2, "0");
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const sec = String(s % 60).padStart(2, "0");
  return `${h}:${m}:${sec}`;
}

function ComplianceGauge({ score }: { score: number }) {
  const r = 52;
  const circ = 2 * Math.PI * r;
  const offset = circ - (score / 100) * circ;
  const colour = score >= 80 ? "#10b981" : score >= 50 ? "#f59e0b" : "#ef4444";
  return (
    <div className="relative flex items-center justify-center w-36 h-36">
      <svg width="144" height="144" viewBox="0 0 144 144" className="absolute">
        <circle cx="72" cy="72" r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="12" />
        <circle cx="72" cy="72" r={r} fill="none" stroke={colour} strokeWidth="12"
          strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
          transform="rotate(-90 72 72)"
          style={{ transition: "stroke-dashoffset 0.6s ease, stroke 0.6s ease" }} />
      </svg>
      <div className="text-center">
        <div className="text-3xl font-black" style={{ color: colour }}>{score}%</div>
        <div className="text-[9px] text-slate-500 uppercase font-bold tracking-widest mt-0.5">Compliance</div>
      </div>
    </div>
  );
}

function StatCard({
  icon, label, value, color,
}: { icon: React.ReactNode; label: string; value: string; color: string }) {
  return (
    <div className="bg-slate-900/60 p-4 rounded-2xl border border-slate-800/80 flex flex-col gap-2">
      <div className="flex items-center gap-1.5 text-slate-600 text-[10px] uppercase font-bold tracking-wider">
        {icon} {label}
      </div>
      <div className={`text-xl font-black ${color}`}>{value}</div>
    </div>
  );
}

export default function Dashboard() {
  /* ── core stream state ── */
  const [stream, setStream] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [latency, setLatency] = useState(0);
  const [fps, setFps] = useState(0);

  /* ── detection state ── */
  const [personCount, setPersonCount] = useState(0);
  const [detectionTags, setDetectionTags] = useState<{ name: string; count: number }[]>([]);
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [criticalEvents, setCriticalEvents] = useState<CriticalEvent[]>([]);
  const [isEmergencyMode, setIsEmergencyMode] = useState(false);
  const [sessionViolations, setSessionViolations] = useState(0);
  const [liveCompliance, setLiveCompliance] = useState(100);

  /* ── system info ── */
  const [deviceInfo, setDeviceInfo] = useState<string>("—");
  const [uptime, setUptime] = useState(0);

  /* ── analytics view ── */
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [stats, setStats] = useState<AnalyticsStats | null>(null);
  const [violationFiles, setViolationFiles] = useState<string[]>([]);

  /* ── refs ── */
  const ws = useRef<WebSocket | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isStreamingRef = useRef(false);
  const frameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(Date.now());
  const pendingFrameRef = useRef(false);
  const inFlightRef = useRef(0);
  const sendTimeRef = useRef(0);
  const sendIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const sessionStartRef = useRef(Date.now());
  const complianceWindowRef = useRef<boolean[]>([]);
  const sessionViolationsRef = useRef(0);

  /* ── uptime ticker ── */
  useEffect(() => {
    const id = setInterval(() => {
      setUptime(Math.floor((Date.now() - sessionStartRef.current) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, []);

  /* ── audio helpers ── */
  const speak = (text: string) => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.9;
    window.speechSynthesis.speak(u);
  };

  const playBeep = useCallback(() => {
    try {
      const ctx = new AudioContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = 880;
      osc.type = "sine";
      gain.gain.setValueAtTime(0.25, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.3);
    } catch { /* user hasn't interacted yet — silent fail is fine */ }
  }, []);

  /* ── WebSocket connection ── */
  useEffect(() => {
    const socket = new WebSocket("ws://127.0.0.1:8000/ws");
    let disposed = false;
    ws.current = socket;
    socket.binaryType = "arraybuffer";

    socket.onopen = () => {
      if (disposed) {
        socket.close();
        return;
      }
      setIsConnected(true);
    };
    socket.onclose = () => {
      if (disposed) return;
      if (ws.current === socket) ws.current = null;
      setIsConnected(false);
      stopWebcam();
    };

    socket.onmessage = (event) => {
      if (disposed) return;
      if (!(event.data instanceof ArrayBuffer)) {
        pendingFrameRef.current = false;
        inFlightRef.current = Math.max(0, inFlightRef.current - 1);
        return;
      }

      /* parse binary frame: [4-byte JSON length][JSON][JPEG] */
      const buf = event.data as ArrayBuffer;
      const view = new DataView(buf);
      const jsonLen = view.getUint32(0);
      const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, jsonLen))) as StreamMetadata;
      const imgBytes = new Uint8Array(buf, 4 + jsonLen);

      /* update annotated frame */
      const blob = new Blob([imgBytes], { type: "image/jpeg" });
      const url = URL.createObjectURL(blob);
      setStream((prev) => { if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev); return url; });

      pendingFrameRef.current = false;
      inFlightRef.current = Math.max(0, inFlightRef.current - 1);
      setLatency(Math.round(performance.now() - sendTimeRef.current));

      /* device */
      if (meta.device) setDeviceInfo(meta.device.toUpperCase());

      /* emergency / critical */
      setIsEmergencyMode(!!meta.response_actions?.emergency_mode);
      if (meta.response_actions?.voice_announcement) speak(meta.response_actions.voice_announcement);
      setCriticalEvents(meta.critical_events?.length ? meta.critical_events : []);

      /* violations */
      const frameViolations = meta.violations ?? [];
      if (meta.alert && frameViolations.length) {
        sessionViolationsRef.current += frameViolations.length;
        setSessionViolations(sessionViolationsRef.current);
        playBeep();
        setAlerts((prev) => [
          { id: Date.now(), time: new Date().toLocaleTimeString(), violations: frameViolations },
          ...prev,
        ].slice(0, 8));
      }

      /* live detections — person count + tag strip */
      const dets: Detection[] = meta.detections ?? [];
      setPersonCount(dets.filter((d) => d.class_name.toLowerCase() === "person").length);
      const tagMap: Record<string, number> = {};
      for (const d of dets) tagMap[d.class_name] = (tagMap[d.class_name] ?? 0) + 1;
      setDetectionTags(Object.entries(tagMap).map(([name, count]) => ({ name, count })));

      /* rolling 60-frame compliance window */
      const hasViolation = frameViolations.length > 0;
      complianceWindowRef.current = [...complianceWindowRef.current.slice(-59), hasViolation];
      const clean = complianceWindowRef.current.filter((v) => !v).length;
      setLiveCompliance(Math.round((clean / complianceWindowRef.current.length) * 100));

      /* fps */
      frameCountRef.current++;
      const now = Date.now();
      if (now - lastFpsTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastFpsTimeRef.current = now;
      }
    };

    return () => {
      disposed = true;
      if (ws.current === socket) ws.current = null;
      if (socket.readyState === WebSocket.OPEN) socket.close();
      if (socket.readyState === WebSocket.CONNECTING) socket.onopen = () => socket.close();
      stopWebcam();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ── analytics fetch ── */
  const fetchAnalytics = useCallback(async () => {
    try {
      const [aRes, vRes] = await Promise.all([
        fetch("http://127.0.0.1:8000/api/analytics"),
        fetch("http://127.0.0.1:8000/api/violations"),
      ]);
      const analyticsData = (await aRes.json()) as AnalyticsStats;
      const violationsData = (await vRes.json()) as ViolationsResponse;
      setStats(analyticsData);
      setViolationFiles(violationsData.violations ?? []);
    } catch { /* backend may be starting up */ }
  }, []);

  useEffect(() => {
    if (!showAnalytics) return;
    fetchAnalytics();
    const id = setInterval(fetchAnalytics, 5000);
    return () => clearInterval(id);
  }, [showAnalytics, fetchAnalytics]);

  /* ── computed analytics values ── */
  const historicalCompliance = (() => {
    if (!stats) return "—";
    const frames = Object.values(stats.daily_trends ?? {}).reduce(
      (s, d) => s + (d.person_frames ?? 0), 0,
    );
    if (frames === 0) return "N/A";
    const rate = (stats.total_violations ?? 0) / frames;
    return `${Math.max(0, Math.min(100, Math.round((1 - rate) * 100)))}%`;
  })();

  const clearGallery = async () => {
    if (!confirm(`Delete all ${violationFiles.length} evidence images? This cannot be undone.`)) return;
    try {
      await fetch("http://127.0.0.1:8000/api/violations", { method: "DELETE" });
      setViolationFiles([]);
    } catch { alert("Failed to clear gallery — is the backend running?"); }
  };

  const downloadCSV = () => {
    if (!stats) return;
    const rows = [["Date", "Person Frames", "Violations"],
      ...Object.entries(stats.daily_trends ?? {}).map(([day, d]) =>
        [day, String(d.person_frames ?? 0), String(d.violations ?? 0)])];
    const blob = new Blob([rows.map((r) => r.join(",")).join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "guardianvision_violations.csv"; a.click();
    URL.revokeObjectURL(url);
  };

  /* ── webcam ── */
  const startWebcam = useCallback(async () => {
    try {
      const ms = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      });
      streamRef.current = ms;
      isStreamingRef.current = true;
      setIsStreaming(true);
      if (videoRef.current) { videoRef.current.srcObject = ms; await videoRef.current.play(); }

      const sendFrame = () => {
        if (!isStreamingRef.current) return;
        if (!videoRef.current || !canvasRef.current || !ws.current) return;
        if (ws.current.readyState !== WebSocket.OPEN) return;
        if (inFlightRef.current >= 2) return; // drop frame if backend is backed up

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        if (ctx && video.readyState === video.HAVE_ENOUGH_DATA) {
          canvas.width = 320; canvas.height = 320;
          ctx.drawImage(video, 0, 0, 320, 320);
          inFlightRef.current++;
          sendTimeRef.current = performance.now();
          canvas.toBlob((blob) => {
            if (blob && ws.current?.readyState === WebSocket.OPEN) ws.current.send(blob);
            else inFlightRef.current = Math.max(0, inFlightRef.current - 1);
          }, "image/jpeg", 0.5);
        }
      };

      sendIntervalRef.current = setInterval(sendFrame, 50); // 20 fps cap
    } catch {
      alert("Could not access webcam. Please check permissions.");
      isStreamingRef.current = false;
      setIsStreaming(false);
    }
  }, []);

  const stopWebcam = useCallback(() => {
    isStreamingRef.current = false;
    setIsStreaming(false);
    if (sendIntervalRef.current) { clearInterval(sendIntervalRef.current); sendIntervalRef.current = null; }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStream(null);
    pendingFrameRef.current = false;
    inFlightRef.current = 0;
  }, []);

  /* ── render ── */
  const emergency = isEmergencyMode;

  return (
    <div
      className={`min-h-screen font-mono transition-colors duration-700 p-6
        ${emergency ? "bg-red-950" : "bg-[#080d16]"} text-slate-100`}
      style={{
        backgroundImage:
          "radial-gradient(circle at 1px 1px, rgba(255,255,255,0.025) 1px, transparent 0)",
        backgroundSize: "28px 28px",
      }}
    >
      <video ref={videoRef} className="hidden" playsInline muted />
      <canvas ref={canvasRef} className="hidden" />

      {/* ── Emergency banner ── */}
      {emergency && (
        <div className="mb-5 flex items-center gap-3 bg-red-500 text-white px-6 py-3 rounded-2xl font-black text-sm animate-pulse shadow-lg shadow-red-900/50">
          <ShieldAlert size={18} />
          EMERGENCY — CRITICAL THREAT DETECTED — INITIATE EVACUATION PROTOCOL
          <ShieldAlert size={18} />
        </div>
      )}

      {/* ── Header ── */}
      <header
        className={`flex flex-wrap justify-between items-center mb-7 pb-5 border-b
          ${emergency ? "border-red-700/60" : "border-slate-800/70"}`}
      >
        <div className="flex items-center gap-4">
          {/* Logo mark */}
          <div
            className={`w-11 h-11 rounded-xl flex items-center justify-center shadow-lg
              ${emergency ? "bg-red-500 shadow-red-900/50" : "bg-blue-600 shadow-blue-900/50"}`}
          >
            <Shield size={22} />
          </div>
          <div>
            <h1
              className={`text-2xl font-black tracking-tight leading-none
                ${emergency ? "text-red-400" : "text-white"}`}
            >
              GUARDIAN VISION
            </h1>
            <p className="text-slate-600 text-[10px] uppercase tracking-[0.18em] mt-1">
              Industrial Safety AI · Visual Compliance Engine
            </p>
          </div>

          {/* Status badges */}
          <div className="flex items-center gap-2 ml-2 flex-wrap">
            <span
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-bold border uppercase
                ${isConnected
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/25"
                  : "bg-rose-500/10 text-rose-400 border-rose-500/25"}`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${isConnected ? "bg-emerald-400 animate-pulse" : "bg-rose-400"}`}
              />
              {isConnected ? "Online" : "Offline"}
            </span>

            {deviceInfo !== "—" && (
              <span className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-[10px] font-bold bg-blue-500/10 text-blue-400 border border-blue-500/25 uppercase">
                <Cpu size={10} /> {deviceInfo}
              </span>
            )}

            <span className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-[10px] font-bold bg-slate-800/80 text-slate-500 border border-slate-700/60">
              <Clock size={10} /> {formatUptime(uptime)}
            </span>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-3 mt-3 sm:mt-0">
          <button
            onClick={isStreaming ? stopWebcam : startWebcam}
            disabled={!isConnected}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-sm border transition-all
              ${isStreaming
                ? "bg-rose-500/15 text-rose-300 border-rose-500/30 hover:bg-rose-500/25"
                : "bg-blue-500/15 text-blue-300 border-blue-500/30 hover:bg-blue-500/25"}
              disabled:opacity-40 disabled:cursor-not-allowed`}
          >
            {isStreaming ? <VideoOff size={15} /> : <Video size={15} />}
            {isStreaming ? "Stop Camera" : "Start Camera"}
          </button>

          <button
            onClick={() => { setShowAnalytics(!showAnalytics); if (!showAnalytics) fetchAnalytics(); }}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-bold text-sm border transition-all
              ${showAnalytics
                ? "bg-violet-500/15 text-violet-300 border-violet-500/30"
                : "bg-slate-800/80 text-slate-400 border-slate-700 hover:bg-slate-700/80"}`}
          >
            <BarChart3 size={15} />
            {showAnalytics ? "Live View" : "Analytics"}
          </button>
        </div>
      </header>

      {/* ════════════════════════════ LIVE VIEW ════════════════════════════ */}
      {!showAnalytics && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Left 2/3 — video + stats */}
          <div className="lg:col-span-2 flex flex-col gap-4">

            {/* Video panel */}
            <div
              className={`relative aspect-video rounded-2xl overflow-hidden border shadow-2xl
                bg-slate-900 flex items-center justify-center
                ${emergency ? "border-red-500/60 shadow-red-950" : "border-slate-800/80 shadow-slate-950"}`}
            >
              {stream ? (
                <>
                  <img src={stream} className="w-full h-full object-cover" alt="Live feed" />
                  {/* CRT scan-line overlay */}
                  <div
                    className="absolute inset-0 pointer-events-none"
                    style={{
                      background:
                        "repeating-linear-gradient(0deg,transparent,transparent 3px,rgba(0,0,0,0.06) 3px,rgba(0,0,0,0.06) 4px)",
                    }}
                  />
                  {/* Corner targeting brackets */}
                  {(["tl","tr","bl","br"] as const).map((c) => (
                    <div key={c} className={`absolute w-6 h-6 border-blue-400/50
                      ${c==="tl" ? "top-3 left-3 border-l-2 border-t-2 rounded-tl" : ""}
                      ${c==="tr" ? "top-3 right-3 border-r-2 border-t-2 rounded-tr" : ""}
                      ${c==="bl" ? "bottom-3 left-3 border-l-2 border-b-2 rounded-bl" : ""}
                      ${c==="br" ? "bottom-3 right-3 border-r-2 border-b-2 rounded-br" : ""}`}
                    />
                  ))}
                </>
              ) : (
                <div className="flex flex-col items-center gap-4 text-slate-700">
                  <Camera size={52} strokeWidth={1} />
                  <p className="text-xs font-bold uppercase tracking-widest">
                    {isConnected ? "Awaiting feed — click Start Camera" : "Backend offline"}
                  </p>
                </div>
              )}

              {/* Live / Standby pill + person count */}
              <div className="absolute top-4 left-4 flex gap-2">
                <span
                  className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-[10px] font-bold border backdrop-blur-sm
                    ${isStreaming
                      ? "bg-black/65 text-red-400 border-red-500/30"
                      : "bg-black/65 text-slate-500 border-slate-700/40"}`}
                >
                  <Radio size={9} className={isStreaming ? "animate-pulse" : ""} />
                  {isStreaming ? "LIVE" : "STANDBY"}
                </span>
                {personCount > 0 && (
                  <span className="flex items-center gap-1.5 px-3 py-1 rounded-lg text-[10px] font-bold bg-black/65 text-cyan-400 border border-cyan-500/30 backdrop-blur-sm">
                    <Users size={9} /> {personCount} {personCount === 1 ? "PERSON" : "PERSONS"}
                  </span>
                )}
              </div>
            </div>

            {/* Detection tag strip */}
            {detectionTags.length > 0 && (
              <div className="flex items-center gap-2 flex-wrap px-1">
                <span className="flex items-center gap-1 text-[10px] text-slate-600 uppercase font-bold">
                  <Eye size={10} /> Detected:
                </span>
                {detectionTags.map(({ name, count }) => (
                  <span
                    key={name}
                    className={`px-2 py-0.5 rounded-md text-[10px] font-bold border
                      ${CLASS_COLOR[name] ?? "bg-slate-800 text-slate-400 border-slate-700"}`}
                  >
                    {name}{count > 1 ? ` ×${count}` : ""}
                  </span>
                ))}
              </div>
            )}

            {/* Stats row */}
            <div className="grid grid-cols-4 gap-3">
              <StatCard icon={<Zap size={13} />}          label="Frame Rate"    value={`${fps} FPS`}          color="text-blue-400" />
              <StatCard icon={<Activity size={13} />}     label="Latency"       value={`${latency}ms`}        color="text-emerald-400" />
              <StatCard icon={<Users size={13} />}        label="In Frame"      value={`${personCount}`}      color="text-cyan-400" />
              <StatCard icon={<AlertTriangle size={13} />} label="Session Alerts" value={`${sessionViolations}`} color="text-rose-400" />
            </div>
          </div>

          {/* Right 1/3 — gauge + events + violations */}
          <div className="flex flex-col gap-4">

            {/* Compliance gauge */}
            <div
              className={`rounded-2xl p-5 border flex flex-col items-center gap-3
                ${emergency ? "bg-red-900/25 border-red-700/40" : "bg-slate-900/60 border-slate-800/80"}`}
            >
              <div className="text-[10px] uppercase font-bold text-slate-600 tracking-widest w-full">
                Live Compliance
              </div>
              <ComplianceGauge score={isStreaming ? liveCompliance : 100} />
              <p className="text-[9px] text-slate-700 text-center">Rolling 60-frame window</p>
            </div>

            {/* Critical events */}
            {criticalEvents.length > 0 && (
              <div className="rounded-2xl p-5 border-2 border-red-500/80 bg-red-950/50 animate-pulse shadow-lg shadow-red-950/50">
                <div className="flex items-center gap-2 mb-3">
                  <ShieldAlert size={15} className="text-red-400" />
                  <span className="text-[11px] font-black uppercase text-red-400 tracking-wider">
                    Critical Threats
                  </span>
                </div>
                <div className="space-y-2">
                  {criticalEvents.map((e, i) => (
                    <div key={i} className="bg-red-900/40 p-3 rounded-xl border border-red-500/25">
                      <div className="font-black text-white text-sm">{e.type}</div>
                      <div className="text-red-300 text-[10px] uppercase font-bold mt-0.5">{e.location}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Active violations */}
            <div
              className={`rounded-2xl p-5 border flex-1 flex flex-col
                ${emergency ? "bg-red-900/20 border-red-700/40" : "bg-slate-900/60 border-slate-800/80"}`}
            >
              <div className="flex items-center gap-2 mb-4">
                <ShieldAlert size={14} className="text-rose-500" />
                <span className="text-[11px] font-black uppercase tracking-wider">Active Violations</span>
                {alerts.length > 0 && (
                  <span className="ml-auto bg-rose-500 text-white text-[10px] font-black px-2 py-0.5 rounded-full">
                    {alerts.length}
                  </span>
                )}
              </div>

              <div className="space-y-2.5 overflow-y-auto flex-1 pr-0.5" style={{ maxHeight: "280px" }}>
                {alerts.length === 0 ? (
                  <div className="flex items-center gap-3 p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/15 text-emerald-500">
                    <CheckCircle size={17} />
                    <span className="text-xs font-semibold">All personnel compliant</span>
                  </div>
                ) : (
                  alerts.map((alert) => (
                    <div key={alert.id}
                      className="bg-rose-500/5 border border-rose-500/20 p-3 rounded-xl">
                      <div className="flex justify-between items-center mb-2">
                        <div className="flex items-center gap-1.5 text-rose-400">
                          <AlertTriangle size={11} />
                          <span className="font-black text-[10px] uppercase">PPE Violation</span>
                        </div>
                        <span className="text-[9px] text-slate-600 font-mono">{alert.time}</span>
                      </div>
                      <div className="flex gap-1.5 flex-wrap">
                        {alert.violations?.[0]?.violations?.map((v: string) => (
                          <span key={v}
                            className="bg-rose-500/20 text-rose-300 px-2 py-0.5 rounded text-[9px] font-black uppercase border border-rose-500/20">
                            No {v}
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
      )}

      {/* ═══════════════════════════ ANALYTICS VIEW ════════════════════════ */}
      {showAnalytics && (
        <div className="space-y-6">

          {/* Top stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard icon={<AlertTriangle size={13} />} label="Total Violations"  value={String(stats?.total_violations ?? "—")} color="text-rose-400" />
            <StatCard icon={<ShieldAlert size={13} />}   label="Hardhat Alerts"    value={String(stats?.violations_by_type?.Hardhat ?? "0")} color="text-amber-400" />
            <StatCard icon={<Shield size={13} />}        label="Vest Alerts"       value={String(stats?.violations_by_type?.["Safety Vest"] ?? "0")} color="text-blue-400" />
            <StatCard icon={<CheckCircle size={13} />}   label="Compliance Score"  value={historicalCompliance} color="text-emerald-400" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

            {/* Evidence gallery */}
            <div className="lg:col-span-2 bg-slate-900/60 rounded-2xl p-6 border border-slate-800/80">
              <div className="flex justify-between items-center mb-5">
                <div className="flex items-center gap-2">
                  <ImageIcon size={14} className="text-blue-400" />
                  <h2 className="text-sm font-black uppercase tracking-wider">Evidence Gallery</h2>
                  {violationFiles.length > 0 && (
                    <span className="bg-blue-500/15 text-blue-400 text-[10px] font-black px-2 py-0.5 rounded-full border border-blue-500/20">
                      {violationFiles.length} captures
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={downloadCSV}
                    className="flex items-center gap-1.5 text-[11px] font-bold text-slate-500 hover:text-white transition-colors px-3 py-1.5 rounded-lg hover:bg-slate-800 border border-transparent hover:border-slate-700"
                  >
                    <Download size={12} /> Export CSV
                  </button>
                  {violationFiles.length > 0 && (
                    <button
                      onClick={clearGallery}
                      className="flex items-center gap-1.5 text-[11px] font-bold text-rose-600 hover:text-rose-400 transition-colors px-3 py-1.5 rounded-lg hover:bg-rose-500/10 border border-transparent hover:border-rose-500/20"
                    >
                      Clear Gallery
                    </button>
                  )}
                </div>
              </div>

              {violationFiles.length === 0 ? (
                <div className="aspect-video flex flex-col items-center justify-center text-slate-700 border-2 border-dashed border-slate-800 rounded-xl">
                  <ImageIcon size={40} className="mb-3 opacity-20" strokeWidth={1} />
                  <p className="text-xs font-bold uppercase tracking-widest opacity-50">No evidence captured yet</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3 max-h-96 overflow-y-auto pr-1">
                  {violationFiles.map((file) => (
                    <div key={file}
                      className="group relative aspect-square bg-slate-800 rounded-xl overflow-hidden border border-slate-700 hover:border-blue-500/50 transition-all">
                      <img
                        src={`http://127.0.0.1:8000/violations/${file}`}
                        className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
                        alt="Violation evidence"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity p-2 flex flex-col justify-end">
                        <p className="text-[8px] font-mono text-slate-300 truncate">{file}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Right column: by-type bars + daily log */}
            <div className="flex flex-col gap-4">

              {/* Violations by type */}
              <div className="bg-slate-900/60 rounded-2xl p-6 border border-slate-800/80">
                <div className="flex items-center gap-2 mb-4">
                  <BarChart3 size={14} className="text-violet-400" />
                  <h2 className="text-sm font-black uppercase tracking-wider">By Type</h2>
                </div>
                {Object.keys(stats?.violations_by_type ?? {}).length === 0 ? (
                  <p className="text-xs text-slate-600 text-center py-6">No violation data yet</p>
                ) : (
                  <div className="space-y-3">
                    {Object.entries(stats?.violations_by_type ?? {}).map(([type, count]) => {
                      const max = Math.max(...Object.values(stats?.violations_by_type ?? {}).map(Number));
                      const pct = max > 0 ? Math.round((count / max) * 100) : 0;
                      return (
                        <div key={type}>
                          <div className="flex justify-between text-[10px] font-bold mb-1">
                            <span className="text-slate-400">{type}</span>
                            <span className="text-rose-400">{count}</span>
                          </div>
                          <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-rose-700 to-rose-400 rounded-full transition-all duration-500"
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Daily log */}
              <div className="bg-slate-900/60 rounded-2xl p-6 border border-slate-800/80 flex-1">
                <div className="flex items-center gap-2 mb-4">
                  <Activity size={14} className="text-emerald-400" />
                  <h2 className="text-sm font-black uppercase tracking-wider">Daily Log</h2>
                </div>
                {Object.keys(stats?.daily_trends ?? {}).length === 0 ? (
                  <p className="text-xs text-slate-600 text-center py-6">No session data yet</p>
                ) : (
                  <div className="space-y-2.5 max-h-52 overflow-y-auto">
                    {Object.entries(stats?.daily_trends ?? {}).reverse().map(([day, data]) => (
                      <div key={day}
                        className="flex justify-between items-center p-3 rounded-xl bg-slate-800/60 border border-slate-700/50">
                        <div>
                          <p className="text-[10px] font-bold text-slate-500">{day}</p>
                          <p className="text-xs font-semibold text-slate-300">{data.person_frames} person-frames</p>
                        </div>
                        <div className="text-right">
                          <p className="text-xs font-black text-rose-400">{data.violations} events</p>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
