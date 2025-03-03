import { ReactNode } from "react";

type NameValueRowProps = {
  name: string;
  value?: ReactNode;
  valueColor?: string;
};

export const NameValueRow: React.FC<NameValueRowProps> = ({
  name,
  value,
  valueColor = "cyan-400",
}) => {
  return (
    <div className="flex flex-row w-full items-baseline text-sm py-1 border-b border-gray-800/30">
      <div className="grow shrink-0 text-gray-500 font-mono tracking-wider text-xs uppercase">
        {name}
      </div>
      <div className={`text-sm shrink text-${valueColor} text-right font-mono tracking-wide digital-flicker`}>
        {value}
      </div>
    </div>
  );
};