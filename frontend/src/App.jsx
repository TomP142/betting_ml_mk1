import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Terminal,
  Database,
  Play,
  AlertCircle,
  ChevronsUp,
  Trophy,
  Activity,
  Wifi,
} from "lucide-react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export default function App() {
  const [logs, setLogs] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [selectedDate, setSelectedDate] = useState(
    new Date().toISOString().split("T")[0],
  );
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [ws, setWs] = useState(null);
  const scrollRef = useRef(null);

  // WebSocket Connection for Logs
  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8000/logs");

    socket.onopen = () => {
      setIsConnected(true);
      addLog("System Connected. Ready for Analysis.");
    };

    socket.onmessage = (event) => {
      addLog(event.data);
    };

    socket.onclose = () => {
      setIsConnected(false);
      addLog("System Disconnected.");
    };

    setWs(socket);

    return () => {
      socket.close();
    };
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const addLog = (msg) => {
    const time = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, `[${time}] ${msg}`]);
  };

  const handleRunAnalysis = async () => {
    if (isLoading) return;
    setIsLoading(true);
    setPredictions([]);
    addLog("Initiating Daily Analysis Sequence...");

    try {
      const response = await fetch("http://localhost:8000/analyze-today", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          date: selectedDate,
          force_train: false,
        }),
      });

      const data = await response.json();
      if (data.results) {
        setPredictions(data.results);
        addLog(
          `Analysis Complete. ${data.results.length} predictions received.`,
        );
      }
    } catch (e) {
      addLog(`CRITICAL ERROR: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 text-gray-200">
      {/* Header */}
      <header className="max-w-7xl mx-auto flex justify-between items-center mb-12">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-neon-green/80 animate-pulse" />
          <h1 className="text-3xl font-bold tracking-tighter text-white">
            BETTING<span className="text-neon-green">.AI</span>{" "}
            <span className="text-xs opacity-50 font-mono border border-white/20 px-2 py-0.5 rounded">
              MK1
            </span>
          </h1>
        </div>
        <div className="flex items-center gap-4 text-sm font-mono text-gray-400">
          <div className="flex items-center gap-2">
            <Wifi
              size={14}
              className={isConnected ? "text-neon-green" : "text-red-500"}
            />
            {isConnected ? "ONLINE" : "OFFLINE"}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Controls & Console */}
        <div className="space-y-6 lg:col-span-1">
          {/* Control Panel */}
          <div className="glass p-6 rounded-2xl relative overflow-hidden group">
            <div className="absolute inset-0 bg-neon-green/5 opacity-0 group-hover:opacity-100 transition-opacity duration-700 pointer-events-none" />

            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Activity className="text-neon-blue" /> Control Center
            </h2>

            <div className="mb-4">
              <label className="block text-xs font-mono text-gray-500 mb-2">
                TARGET DATE
              </label>
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-2 text-white font-mono focus:border-neon-green/50 outline-none"
              />
            </div>

            <button
              onClick={handleRunAnalysis}
              disabled={isLoading || !isConnected}
              className="w-full h-14 bg-gradient-to-r from-neon-blue to-neon-purple rounded-xl font-bold text-white shadow-lg shadow-neon-blue/20 hover:shadow-neon-blue/40 transform hover:scale-[1.02] active:scale-[0.98] transition-all flex items-center justify-center gap-3 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  ANALYZING...
                </>
              ) : (
                <>
                  <Play fill="currentColor" /> FIND TODAY'S BETS
                </>
              )}
            </button>

            <p className="mt-4 text-xs text-gray-500 text-center">
              Scans live schedule, trains models, and calculates variance.
            </p>
          </div>

          {/* Console Log */}
          <div className="glass p-4 rounded-2xl h-[500px] flex flex-col font-mono text-xs relative overflow-hidden">
            <div className="flex items-center justify-between mb-2 pb-2 border-b border-white/5">
              <span className="flex items-center gap-2 text-gray-400">
                <Terminal size={14} /> SYSTEM LOG
              </span>
              <span className="w-2 h-2 rounded-full bg-neon-green animate-blink" />
            </div>

            <div
              ref={scrollRef}
              className="flex-1 overflow-y-auto space-y-1 scrollbar-hide text-green-400/80"
            >
              {logs.length === 0 && (
                <span className="opacity-30">Waiting for command...</span>
              )}
              {logs.map((log, i) => {
                const splitIndex = log.indexOf("]");
                const timestamp = log.substring(0, splitIndex + 1);
                const content = log.substring(splitIndex + 1);
                return (
                  <div key={i} className="break-words">
                    <span className="opacity-50 mr-2">{timestamp}</span>
                    {content}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Results */}
        <div className="lg:col-span-2 space-y-6">
          <h2 className="text-xl font-bold flex items-center gap-2">
            <Trophy className="text-amber-400" /> TOP PICKS
          </h2>

          {predictions.length === 0 ? (
            <div className="h-[400px] border-2 border-dashed border-white/10 rounded-3xl flex items-center justify-center text-gray-600">
              NO DATA AVAILABLE
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Top 4 Cards */}
              {predictions.slice(0, 50).map((pred, i) => (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  key={i}
                  className="glass-card p-5 rounded-2xl relative overflow-hidden group hover:border-neon-green/30 transition-colors"
                >
                  {/* Header */}
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="text-xl font-bold text-white">
                        {pred.PLAYER_NAME}
                      </h3>
                      <div className="text-xs text-gray-400 flex items-center gap-2 mt-1">
                        <span
                          className={cn(
                            "px-1.5 py-0.5 rounded font-bold text-[10px]",
                            pred.IS_HOME
                              ? "bg-white/10 text-white"
                              : "bg-neon-blue/20 text-neon-blue",
                          )}
                        >
                          {pred.IS_HOME ? "HOME" : "AWAY"}
                        </span>
                        <span className="bg-white/5 border border-white/10 px-1.5 py-0.5 rounded text-[10px]">
                          {pred.MATCHUP}
                        </span>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-gray-400">Proj. PRA</div>
                      <div className="text-2xl font-black text-neon-green tracking-tighter">
                        {pred.PRED_PRA?.toFixed(1) ||
                          (
                            pred.PRED_PTS +
                            pred.PRED_REB +
                            pred.PRED_AST
                          ).toFixed(1)}
                      </div>
                    </div>
                  </div>

                  {/* Main Stats Grid */}
                  <div className="grid grid-cols-3 gap-2 mt-4 bg-black/40 p-3 rounded-xl border border-white/5">
                    <div className="text-center">
                      <div className="text-[10px] text-gray-500 mb-1 uppercase tracking-wider">
                        Points
                      </div>
                      <div className="font-bold text-lg">
                        {pred.PRED_PTS.toFixed(1)}
                      </div>
                      <div className="text-[10px] text-gray-600 mt-1">
                        Line: {pred.LINE_PTS_LOW?.toFixed(1)} -{" "}
                        {pred.LINE_PTS_HIGH?.toFixed(1)}
                      </div>
                    </div>
                    <div className="text-center border-l border-white/5">
                      <div className="text-[10px] text-gray-500 mb-1 uppercase tracking-wider">
                        Rebs
                      </div>
                      <div className="font-bold text-lg">
                        {pred.PRED_REB.toFixed(1)}
                      </div>
                      <div className="text-[10px] text-gray-600 mt-1">
                        Line: {pred.LINE_REB_LOW?.toFixed(1)} -{" "}
                        {pred.LINE_REB_HIGH?.toFixed(1)}
                      </div>
                    </div>
                    <div className="text-center border-l border-white/5">
                      <div className="text-[10px] text-gray-500 mb-1 uppercase tracking-wider">
                        Asts
                      </div>
                      <div className="font-bold text-lg">
                        {pred.PRED_AST.toFixed(1)}
                      </div>
                      <div className="text-[10px] text-gray-600 mt-1">
                        Line: {pred.LINE_AST_LOW?.toFixed(1)} -{" "}
                        {pred.LINE_AST_HIGH?.toFixed(1)}
                      </div>
                    </div>
                  </div>

                  {/* Combos & Detailed Lines */}
                  <div className="mt-4 pt-4 border-t border-white/5 grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-[10px] font-mono text-gray-500 mb-2 uppercase">
                        Combos
                      </div>
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-400">Pts+Reb</span>
                          <span className="font-bold text-white">
                            {pred.PRED_PR?.toFixed(1)}
                          </span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-400">Pts+Ast</span>
                          <span className="font-bold text-white">
                            {pred.PRED_PA?.toFixed(1)}
                          </span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-400">Reb+Ast</span>
                          <span className="font-bold text-white">
                            {pred.PRED_RA?.toFixed(1)}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <div className="text-[10px] font-mono text-gray-500 mb-2 uppercase flex items-center gap-1">
                        <AlertCircle size={10} /> Safe Zones
                      </div>
                      <div className="space-y-1.5 text-[10px] text-gray-400">
                        <div className="flex justify-between bg-white/5 px-2 py-1 rounded">
                          <span>Bet Over</span>
                          <span className="text-neon-green font-mono font-bold">
                            &lt; {pred.LINE_PRA_LOW?.toFixed(1)}
                          </span>
                        </div>
                        <div className="flex justify-between bg-white/5 px-2 py-1 rounded">
                          <span>Bet Under</span>
                          <span className="text-red-400 font-mono font-bold">
                            &gt; {pred.LINE_PRA_HIGH?.toFixed(1)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}

          {/* Full Table Link or Expander could go here */}
        </div>
      </main>
    </div>
  );
}
