import React from "react";
import { Switch, Route } from "wouter";
import { Toaster } from "@/components/ui/toaster";
import { DashboardShell } from "./components/layout/DashboardShell";
import NotFound from "@/pages/not-found";
import Overview from "./pages/overview";
import MemoryCore from "./pages/memory-core";
import NeuralMemory from "./pages/neural-memory";
import CCE from "./pages/cce";
import AssembliesIndex from "./pages/assemblies/index";
import AssemblyDetail from "./pages/assemblies/[id]";
import LLMGuidance from "./pages/llm-guidance";
import Logs from "./pages/logs";
import Chat from "./pages/chat";
import Config from "./pages/config";
import Admin from "./pages/admin";
import { useEffect } from "react";
import { usePollingStore } from "./lib/store";
import { FeaturesProvider } from "./contexts/FeaturesContext";
import Phase59Tester from "./components/debug/Phase59Tester";

function Router() {
  const { startPolling, stopPolling } = usePollingStore();

  // Start the polling when the app loads
  useEffect(() => {
    startPolling();
    
    // Cleanup on unmount
    return () => {
      stopPolling();
    };
  }, [startPolling, stopPolling]);

  return (
    <DashboardShell>
      <Switch>
        <Route path="/" component={Overview} />
        <Route path="/overview" component={Overview} />
        <Route path="/memory-core" component={MemoryCore} />
        <Route path="/neural-memory" component={NeuralMemory} />
        <Route path="/cce" component={CCE} />
        <Route path="/assemblies" component={AssembliesIndex} />
        <Route path="/assemblies/:id" component={AssemblyDetail} />
        <Route path="/llm-guidance" component={LLMGuidance} />
        <Route path="/logs" component={Logs} />
        <Route path="/chat" component={Chat} />
        <Route path="/config" component={Config} />
        <Route path="/admin" component={Admin} />
        <Route path="/debug/phase59" component={Phase59Tester} />
        <Route component={NotFound} />
      </Switch>
    </DashboardShell>
  );
}

function App() {
  return (
    <FeaturesProvider>
      <Router />
      <Toaster />
    </FeaturesProvider>
  );
}

export default App;
