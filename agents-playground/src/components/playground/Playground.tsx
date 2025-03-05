"use client";

import { LoadingSVG } from "@/components/button/LoadingSVG";
import { ChatMessageType } from "@/components/chat/ChatTile";
import { ColorPicker } from "@/components/colorPicker/ColorPicker";
import { AudioInputTile } from "@/components/config/AudioInputTile";
import { ConfigurationPanelItem } from "@/components/config/ConfigurationPanelItem";
import { NameValueRow } from "@/components/config/NameValueRow";
import { PlaygroundHeader } from "@/components/playground/PlaygroundHeader";
import { MemoryDashboardTile } from '@/components/memory/MemoryDashboardTile';
import {
  PlaygroundTab,
  PlaygroundTabbedTile,
  PlaygroundTile,
} from "@/components/playground/PlaygroundTile";
import { useConfig } from "@/hooks/useConfig";
import { TranscriptionTile } from "@/transcriptions/TranscriptionTile";
import {
  BarVisualizer,
  VideoTrack,
  useConnectionState,
  useDataChannel,
  useLocalParticipant,
  useRoomInfo,
  useTracks,
  useVoiceAssistant,
} from "@livekit/components-react";
import { ConnectionState, LocalParticipant, Track } from "livekit-client";
import { QRCodeSVG } from "qrcode.react";
import { ReactNode, useCallback, useEffect, useMemo, useState, useRef } from "react";
import tailwindTheme from "../../lib/tailwindTheme.preval";
import { createNeuralParticles, createGlitchEffect } from "@/lib/animations";
import { UserSettings } from "@/hooks/useConfig";

export interface PlaygroundMeta {
  name: string;
  value: string;
}

export interface PlaygroundProps {
  logo?: ReactNode;
  themeColors: string[];
  onConnect: (connect: boolean, opts?: { token: string; url: string }) => void;
}

const headerHeight = 56;

export default function Playground({
  logo,
  themeColors,
  onConnect,
}: PlaygroundProps) {
  const { config, setUserSettings } = useConfig();
  const { name } = useRoomInfo();
  const [transcripts, setTranscripts] = useState<ChatMessageType[]>([]);
  const { localParticipant } = useLocalParticipant();
  const playgroundRef = useRef<HTMLDivElement>(null);

  const voiceAssistant = useVoiceAssistant();

  const roomState = useConnectionState();
  const tracks = useTracks();

  useEffect(() => {
    if (roomState === ConnectionState.Connected) {
      localParticipant.setCameraEnabled(config.settings.inputs.camera);
      localParticipant.setMicrophoneEnabled(config.settings.inputs.mic);
    }
  }, [config, localParticipant, roomState]);

  useEffect(() => {
    if (playgroundRef.current) {
      // Add neural particle effects to the playground
      const cleanupParticles = createNeuralParticles(playgroundRef.current, 15);
      
      // Add occasional glitch effect
      const cleanupGlitch = createGlitchEffect(playgroundRef.current, 0.5);
      
      return () => {
        cleanupParticles();
        cleanupGlitch();
      };
    }
  }, []);

  const agentVideoTrack = tracks.find(
    (trackRef) =>
      trackRef.publication.kind === Track.Kind.Video &&
      trackRef.participant.isAgent
  );

  const localTracks = tracks.filter(
    ({ participant }) => participant instanceof LocalParticipant
  );
  const localVideoTrack = localTracks.find(
    ({ source }) => source === Track.Source.Camera
  );
  const localMicTrack = localTracks.find(
    ({ source }) => source === Track.Source.Microphone
  );

  const onDataReceived = useCallback(
    (msg: any) => {
      if (msg.topic === "transcription") {
        const decoded = JSON.parse(
          new TextDecoder("utf-8").decode(msg.payload)
        );
        let timestamp = new Date().getTime();
        if ("timestamp" in decoded && decoded.timestamp > 0) {
          timestamp = decoded.timestamp;
        }
        setTranscripts([
          ...transcripts,
          {
            name: "You",
            message: decoded.text,
            timestamp: timestamp,
            isSelf: true,
          },
        ]);
      }
    },
    [transcripts]
  );

  useDataChannel(onDataReceived);

  const videoTileContent = useMemo(() => {
    const videoFitClassName = `object-${config.video_fit || "cover"}`;

    const disconnectedContent = (
      <div className="flex items-center justify-center text-cyan-500 text-center w-full h-full font-mono tracking-wide opacity-70">
        <div className="glass-panel p-6 border-cyan-500/20">
          No video track. Connect to get started.
        </div>
      </div>
    );

    const loadingContent = (
      <div className="flex flex-col items-center gap-4 text-cyan-400 text-center h-full w-full font-mono tracking-wider">
        <LoadingSVG />
        <div className="digital-flicker">Initializing neural interface...</div>
      </div>
    );

    const videoContent = (
      <div className="relative w-full h-full">
        <VideoTrack
          trackRef={agentVideoTrack}
          className={`absolute top-1/2 -translate-y-1/2 ${videoFitClassName} object-position-center w-full h-full`}
        />
        
        {/* Video overlay elements for cyberpunk effect */}
        <div className="absolute inset-0 pointer-events-none border border-cyan-500/20"></div>
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-cyan-500/40 to-transparent"></div>
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-cyan-500/40 to-transparent"></div>
        
        {/* Status indicators */}
        <div className="absolute top-4 right-4 flex items-center gap-2">
          <div className="text-xs font-mono text-cyan-400 tracking-wider bg-black/40 px-2 py-1 rounded-sm">
            NEURAL LINK: <span className="text-cyan-300 status-active">ACTIVE</span>
          </div>
        </div>
        
        {/* Scan line effect */}
        <div className="absolute inset-0 scan-line pointer-events-none"></div>
      </div>
    );

    let content = null;
    if (roomState === ConnectionState.Disconnected) {
      content = disconnectedContent;
    } else if (agentVideoTrack) {
      content = videoContent;
    } else {
      content = loadingContent;
    }

    return (
      <div className="flex flex-col w-full grow text-cyan-500 bg-black/50 rounded-sm border border-gray-800 relative overflow-hidden">
        {content}
      </div>
    );
  }, [agentVideoTrack, config, roomState]);

  useEffect(() => {
    document.body.style.setProperty(
      "--lk-theme-color",
      // @ts-ignore
      tailwindTheme.colors[config.settings.theme_color]["500"]
    );
    document.body.style.setProperty(
      "--lk-drop-shadow",
      `var(--lk-theme-color) 0px 0px 18px`
    );
  }, [config.settings.theme_color]);

  const audioTileContent = useMemo(() => {
    const disconnectedContent = (
      <div className="flex flex-col items-center justify-center gap-2 text-cyan-500 text-center w-full font-mono tracking-wide">
        <div className="glass-panel p-6 border-cyan-500/20">
          No audio track. Connect to get started.
        </div>
      </div>
    );

    const waitingContent = (
      <div className="flex flex-col items-center gap-4 text-cyan-400 text-center w-full font-mono tracking-wider">
        <LoadingSVG />
        <div className="digital-flicker">Initializing audio processor...</div>
      </div>
    );

    const visualizerContent = (
      <div
        className={`flex items-center justify-center w-full h-48 [--lk-va-bar-width:30px] [--lk-va-bar-gap:20px] [--lk-fg:var(--lk-theme-color)] relative`}
      >
        {/* Custom visualizer wrapper for cyberpunk effect */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-full max-w-md h-full flex items-center justify-center relative border border-cyan-500/20 bg-black/30 rounded-sm p-4">
            <BarVisualizer
              state={voiceAssistant.state}
              trackRef={voiceAssistant.audioTrack}
              barCount={5}
              options={{ minHeight: 20 }}
            />
            
            {/* Neural pathway particles */}
            <div className="absolute inset-0 particles-container"></div>
            
            {/* Audio status indicator */}
            <div className="absolute bottom-2 right-2 text-xs font-mono text-cyan-400 digital-flicker tracking-wider">
              AUDIO SIGNAL: ACTIVE
            </div>
          </div>
        </div>
      </div>
    );

    if (roomState === ConnectionState.Disconnected) {
      return disconnectedContent;
    }

    if (!voiceAssistant.audioTrack) {
      return waitingContent;
    }

    return visualizerContent;
  }, [
    voiceAssistant.audioTrack,
    config.settings.theme_color,
    roomState,
    voiceAssistant.state,
  ]);

  const chatTileContent = useMemo(() => {
    if (voiceAssistant.audioTrack) {
      return (
        <TranscriptionTile
          agentAudioTrack={voiceAssistant.audioTrack}
          accentColor={config.settings.theme_color}
        />
      );
    }
    return <></>;
  }, [config.settings.theme_color, voiceAssistant.audioTrack]);

  // Update theme color setting
  const updateThemeColor = useCallback((color: string) => {
    const newSettings = { ...config.settings };
    newSettings.theme_color = color;
    setUserSettings(newSettings);
  }, [config.settings, setUserSettings]);

  const settingsTileContent = useMemo(() => {
    return (
      <div className="flex flex-col gap-3 h-full overflow-y-auto pr-2">
        {config.settings.memory_enabled && (
          <div className="w-full">
            <MemoryDashboardTile
              accentColor={config.settings.theme_color}
            />
          </div>
        )}
        {/* Debug info */}
        <div className="text-xs text-gray-400 mb-2">
          Memory enabled: {config.settings.memory_enabled !== undefined ? String(config.settings.memory_enabled) : "false"}
        </div>
        {config.description && (
          <ConfigurationPanelItem title="System Overview">
            <div className="font-mono text-cyan-300/90 text-xs leading-relaxed tracking-wide">
              {config.description}
            </div>
          </ConfigurationPanelItem>
        )}

        <ConfigurationPanelItem title="Neural Interface">
          {localParticipant && (
            <div className="flex flex-col gap-2">
              <NameValueRow
                name="Interface ID"
                value={name}
                valueColor={`${config.settings.theme_color}-400`}
              />
              <NameValueRow
                name="User Identity"
                value={localParticipant.identity}
              />
              <NameValueRow
                name="Connection Status"
                value={
                  <span className="status-active">SYNCHRONIZED</span>
                }
              />
            </div>
          )}
        </ConfigurationPanelItem>
        
        <ConfigurationPanelItem title="System Status">
          <div className="flex flex-col gap-2">
            <NameValueRow
              name="Neural Link"
              value={
                roomState === ConnectionState.Connecting ? (
                  <LoadingSVG diameter={16} strokeWidth={2} />
                ) : (
                  roomState.toUpperCase()
                )
              }
              valueColor={
                roomState === ConnectionState.Connected
                  ? `${config.settings.theme_color}-400`
                  : "gray-500"
              }
            />
            <NameValueRow
              name="AI Core"
              value={
                voiceAssistant.agent ? (
                  "ACTIVE"
                ) : roomState === ConnectionState.Connected ? (
                  <LoadingSVG diameter={12} strokeWidth={2} />
                ) : (
                  "INACTIVE"
                )
              }
              valueColor={
                voiceAssistant.agent
                  ? `${config.settings.theme_color}-400`
                  : "gray-500"
              }
            />
            <NameValueRow
              name="Quantum Database"
              value="ONLINE"
              valueColor="amber-400"
            />
          </div>
        </ConfigurationPanelItem>
        
        {localVideoTrack && (
          <ConfigurationPanelItem
            title="Visual Input"
            deviceSelectorKind="videoinput"
          >
            <div className="relative border border-cyan-500/20 rounded-sm overflow-hidden">
              <VideoTrack
                className="rounded-sm w-full cyber-loading"
                trackRef={localVideoTrack}
              />
            </div>
          </ConfigurationPanelItem>
        )}
        
        {config.show_qr && (
          <div className="w-full">
            <ConfigurationPanelItem title="Mobile Sync">
              <div className="p-2 bg-white w-fit mx-auto">
                <QRCodeSVG value={window.location.href} width="128" />
              </div>
              <div className="text-xs text-center mt-2 text-cyan-400">
                Scan to synchronize with mobile device
              </div>
            </ConfigurationPanelItem>
          </div>
        )}

        {/* Moved Audio Input up from bottom */}
        {localMicTrack && (
          <ConfigurationPanelItem
            title="Audio Input"
            deviceSelectorKind="audioinput"
          >
            <AudioInputTile trackRef={localMicTrack} />
          </ConfigurationPanelItem>
        )}
        
        {/* Moved Theme Selection to bottom */}
        <div className="w-full mt-auto">
          <ConfigurationPanelItem title="Interface Theme">
            <ColorPicker
              colors={themeColors}
              selectedColor={config.settings.theme_color}
              onSelect={updateThemeColor}
            />
          </ConfigurationPanelItem>
        </div>
      </div>
    );
  }, [
    config.description,
    config.settings,
    config.show_qr,
    localParticipant,
    name,
    roomState,
    localVideoTrack,
    localMicTrack,
    themeColors,
    updateThemeColor,
    voiceAssistant.agent,
  ]);

  let mobileTabs: PlaygroundTab[] = [];
  if (config.settings.outputs.video) {
    mobileTabs.push({
      title: "Neural Interface",
      content: (
        <PlaygroundTile
          className="w-full h-full grow"
          childrenClassName="justify-center"
        >
          {videoTileContent}
        </PlaygroundTile>
      ),
    });
  }

  if (config.settings.outputs.audio) {
    mobileTabs.push({
      title: "Audio Analysis",
      content: (
        <PlaygroundTile
          className="w-full h-full grow"
          childrenClassName="justify-center"
        >
          {audioTileContent}
        </PlaygroundTile>
      ),
    });
  }

  if (config.settings.chat) {
    mobileTabs.push({
      title: "Neural Chat",
      content: chatTileContent,
    });
  }

  mobileTabs.push({
    title: "System Controls",
    content: (
      <PlaygroundTile
        padding={false}
        backgroundColor="gray-950"
        className="h-full w-full basis-1/4 items-start overflow-y-auto flex"
        childrenClassName="h-full grow items-start"
      >
        {settingsTileContent}
      </PlaygroundTile>
    ),
  });

  return (
    <>
      <PlaygroundHeader
        title={
          <span className="text-cyan-400 tracking-wider text-glow digital-flicker font-mono">
            SYNTHIENCE.AI <span className="text-xs">v2.0</span>
          </span>
        }
        logo={logo}
        githubLink={config.github_link}
        height={headerHeight}
        accentColor={config.settings.theme_color}
        connectionState={roomState}
        onConnectClicked={() =>
          onConnect(roomState === ConnectionState.Disconnected)
        }
      />
      <div
        ref={playgroundRef}
        className={`flex gap-4 p-4 grow w-full selection:bg-${config.settings.theme_color}-900 holo-grid relative`}
        style={{ 
          height: `calc(100% - ${headerHeight}px)`,
          maxHeight: `calc(100% - ${headerHeight}px)`,
          overflow: 'hidden'
        }}
      >
        {/* Mobile layout */}
        <div className="flex flex-col gap-4 h-full w-full lg:hidden">
          <PlaygroundTabbedTile
            className="h-full w-full"
            tabs={mobileTabs}
            initialTab={0}
          />
        </div>

        {/* Desktop layout */}
        <div className="hidden lg:flex h-full w-full gap-4">
          {/* Left side: Video/Audio */}
          <div className={`flex flex-col grow basis-1/2 gap-4 h-full ${
            !config.settings.outputs.audio && !config.settings.outputs.video
              ? "hidden"
              : "flex"
          }`}>
            {config.settings.outputs.video && (
              <PlaygroundTile
                title="Neural Interface"
                className="w-full flex-1"
                childrenClassName="justify-center"
              >
                {videoTileContent}
              </PlaygroundTile>
            )}
            {config.settings.outputs.audio && (
              <PlaygroundTile
                title="Audio Analysis"
                className="w-full flex-1"
                childrenClassName="justify-center"
              >
                {audioTileContent}
              </PlaygroundTile>
            )}
          </div>

          {/* Middle: Chat */}
          {config.settings.chat && (
            <PlaygroundTile
              title="Neural Chat"
              className="h-full grow basis-1/4"
            >
              {chatTileContent}
            </PlaygroundTile>
          )}

          {/* Right side: Settings/Controls */}
          <PlaygroundTile
            padding={false}
            backgroundColor="gray-950"
            className="h-full w-full basis-1/4 max-w-[400px] overflow-hidden"
            childrenClassName="h-full"
          >
            {settingsTileContent}
          </PlaygroundTile>
        </div>
      </div>
    </>
  );
}