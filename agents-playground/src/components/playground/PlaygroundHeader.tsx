import { Button } from "@/components/button/Button";
import { LoadingSVG } from "@/components/button/LoadingSVG";
import { SettingsDropdown } from "@/components/playground/SettingsDropdown";
import { useConfig } from "@/hooks/useConfig";
import { ConnectionState } from "livekit-client";
import { ReactNode, useEffect, useRef } from "react";
import { createTextFlicker } from "@/lib/animations";

type PlaygroundHeaderProps = {
  logo?: ReactNode;
  title?: ReactNode;
  githubLink?: string;
  height: number;
  accentColor: string;
  connectionState: ConnectionState;
  onConnectClicked: () => void;
};

export const PlaygroundHeader = ({
  logo,
  title,
  githubLink,
  accentColor,
  height,
  onConnectClicked,
  connectionState,
}: PlaygroundHeaderProps) => {
  const { config } = useConfig();
  const titleRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (titleRef.current) {
      // Add text flicker effect to the title
      createTextFlicker(titleRef.current);
    }
  }, []);

  return (
    <div
      className={`
        flex gap-4 pt-4 text-${accentColor}-500 justify-between items-center shrink-0
        glass-panel border-b border-${accentColor}-800/30 backdrop-blur-md
      `}
      style={{
        height: height + "px",
      }}
    >
      <div className="flex items-center gap-3 basis-2/3">
        <div className="flex lg:basis-1/2">
          <a href="https://livekit.io" className="hover:opacity-80 transition-opacity">
            {logo ?? <SynthienceLogo accentColor={accentColor} />}
          </a>
        </div>
        
        <div 
          ref={titleRef}
          className="lg:basis-1/2 lg:text-center text-xs lg:text-base lg:font-medium text-cyan-400 digital-flicker tracking-wider"
        >
          {title}
        </div>
      </div>
      
      <div className="flex basis-1/3 justify-end items-center gap-3">
        {githubLink && (
          <a
            href={githubLink}
            target="_blank"
            className={`text-white hover:text-${accentColor}-300 transition-colors duration-300`}
            title="View on GitHub"
          >
            <GithubSVG />
          </a>
        )}
        
        {config.settings.editable && <SettingsDropdown />}
        
        <Button
          accentColor={
            connectionState === ConnectionState.Connected ? "red" : accentColor
          }
          disabled={connectionState === ConnectionState.Connecting}
          onClick={() => {
            onConnectClicked();
          }}
        >
          {connectionState === ConnectionState.Connecting ? (
            <LoadingSVG />
          ) : connectionState === ConnectionState.Connected ? (
            "Disconnect"
          ) : (
            "Initialize"
          )}
        </Button>
      </div>
    </div>
  );
};

const SynthienceLogo = ({ accentColor }: { accentColor: string }) => (
  <svg
    width="32"
    height="32"
    viewBox="0 0 32 32"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={`text-${accentColor}-500`}
  >
    <path
      d="M16 0C7.163 0 0 7.163 0 16C0 24.837 7.163 32 16 32C24.837 32 32 24.837 32 16C32 7.163 24.837 0 16 0ZM16 2C23.732 2 30 8.268 30 16C30 23.732 23.732 30 16 30C8.268 30 2 23.732 2 16C2 8.268 8.268 2 16 2Z"
      fill="currentColor"
    />
    <path
      d="M16 6C10.477 6 6 10.477 6 16C6 21.523 10.477 26 16 26C21.523 26 26 21.523 26 16C26 10.477 21.523 6 16 6ZM16 8C20.418 8 24 11.582 24 16C24 20.418 20.418 24 16 24C11.582 24 8 20.418 8 16C8 11.582 11.582 8 16 8Z"
      fill="currentColor"
    />
    <path
      d="M16 12C13.791 12 12 13.791 12 16C12 18.209 13.791 20 16 20C18.209 20 20 18.209 20 16C20 13.791 18.209 12 16 12Z"
      fill="currentColor"
    />
    <path
      d="M24 2H30V8"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
    />
    <path
      d="M8 30H2V24"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
    />
  </svg>
);

const GithubSVG = () => (
  <svg
    width="24"
    height="24"
    viewBox="0 0 98 96"
    xmlns="http://www.w3.org/2000/svg"
  >
    <path
      fillRule="evenodd"
      clipRule="evenodd"
      d="M48.854 0C21.839 0 0 22 0 49.217c0 21.756 13.993 40.172 33.405 46.69 2.427.49 3.316-1.059 3.316-2.362 0-1.141-.08-5.052-.08-9.127-13.59 2.934-16.42-5.867-16.42-5.867-2.184-5.704-5.42-7.17-5.42-7.17-4.448-3.015.324-3.015.324-3.015 4.934.326 7.523 5.052 7.523 5.052 4.367 7.496 11.404 5.378 14.235 4.074.404-3.178 1.699-5.378 3.074-6.6-10.839-1.141-22.243-5.378-22.243-24.283 0-5.378 1.94-9.778 5.014-13.2-.485-1.222-2.184-6.275.486-13.038 0 0 4.125-1.304 13.426 5.052a46.97 46.97 0 0 1 12.214-1.63c4.125 0 8.33.571 12.213 1.63 9.302-6.356 13.427-5.052 13.427-5.052 2.67 6.763.97 11.816.485 13.038 3.155 3.422 5.015 7.822 5.015 13.2 0 18.905-11.404 23.06-22.324 24.283 1.78 1.548 3.316 4.481 3.316 9.126 0 6.6-.08 11.897-.08 13.526 0 1.304.89 2.853 3.316 2.364 19.412-6.52 33.405-24.935 33.405-46.691C97.707 22 75.788 0 48.854 0z"
      fill="currentColor"
    />
  </svg>
);