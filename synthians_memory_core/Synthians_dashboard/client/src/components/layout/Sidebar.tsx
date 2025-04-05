import React from "react";
import { Link, useLocation } from "wouter";
import { cn } from "@/lib/utils";

type NavLinkProps = {
  href: string;
  icon: string;
  label: string;
  isActive: boolean;
};

const NavLink = ({ href, icon, label, isActive }: NavLinkProps) => {
  return (
    <Link href={href}>
      <div className={cn(
        "flex items-center px-4 py-2 text-sm hover:bg-muted hover:text-foreground cursor-pointer",
        isActive 
          ? "text-primary bg-muted border-l-2 border-primary" 
          : "text-gray-400"
      )}>
        <i className={`fas fa-${icon} w-5`}></i>
        <span>{label}</span>
      </div>
    </Link>
  );
};

const NavGroup = ({ title, children }: { title: string, children: React.ReactNode }) => {
  return (
    <>
      <div className="px-4 py-2 mt-4 text-xs text-gray-500 uppercase">{title}</div>
      {children}
    </>
  );
};

export function Sidebar() {
  const [location] = useLocation();

  return (
    <aside className="w-64 border-r border-border bg-sidebar hidden md:block">
      {/* Logo */}
      <div className="flex items-center p-4 border-b border-border">
        <div className="w-8 h-8 rounded-md bg-gradient-to-br from-primary to-accent flex items-center justify-center mr-2">
          <span className="text-white font-bold">S</span>
        </div>
        <div>
          <h1 className="text-lg font-bold text-primary">Synthians</h1>
          <p className="text-xs text-gray-500">Cognitive Dashboard v1.0</p>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="py-4">
        <NavGroup title="Monitoring">
          <NavLink 
            href="/" 
            icon="tachometer-alt" 
            label="System Overview" 
            isActive={location === "/"} 
          />
          <NavLink 
            href="/memory-core" 
            icon="database" 
            label="Memory Core" 
            isActive={location === "/memory-core"} 
          />
          <NavLink 
            href="/neural-memory" 
            icon="brain" 
            label="Neural Memory" 
            isActive={location === "/neural-memory"} 
          />
          <NavLink 
            href="/cce" 
            icon="sitemap" 
            label="CCE" 
            isActive={location === "/cce"} 
          />
        </NavGroup>

        <NavGroup title="Tools">
          <NavLink 
            href="/assemblies" 
            icon="puzzle-piece" 
            label="Assembly Inspector" 
            isActive={Boolean(location && location.indexOf("/assemblies") === 0)} 
          />
          <NavLink 
            href="/llm-guidance" 
            icon="comment" 
            label="LLM Guidance" 
            isActive={location === "/llm-guidance"} 
          />
          <NavLink 
            href="/logs" 
            icon="terminal" 
            label="Logs" 
            isActive={location === "/logs"} 
          />
          <NavLink 
            href="/chat" 
            icon="comments" 
            label="Chat Interface" 
            isActive={location === "/chat"} 
          />
        </NavGroup>

        <NavGroup title="Settings">
          <NavLink 
            href="/config" 
            icon="cog" 
            label="Configuration" 
            isActive={location === "/config"} 
          />
          <NavLink 
            href="/admin" 
            icon="wrench" 
            label="Admin Actions" 
            isActive={location === "/admin"} 
          />
        </NavGroup>
      </nav>
      
      {/* Status Indicator */}
      <div className="absolute bottom-0 w-64 p-4 border-t border-border">
        <div className="flex items-center">
          <div className="w-2 h-2 rounded-full bg-secondary pulse mr-2"></div>
          <span className="text-xs text-gray-400">All Systems Operational</span>
        </div>
        <div className="mt-2 text-xs text-gray-500">Last updated: 1 minute ago</div>
      </div>
    </aside>
  );
}
