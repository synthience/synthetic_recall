import { LogLevel } from '@/lib/websocket';
import { Button } from '@/components/ui/button';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { Badge } from '@/components/ui/badge';
import { BsInfoCircle, BsExclamationTriangle, BsExclamationCircle, BsTerminal } from 'react-icons/bs';

type LogLevelOrAll = LogLevel | 'all';

interface LogLevelFilterProps {
  selectedLevel: LogLevelOrAll;
  onChange: (level: LogLevelOrAll) => void;
  counts?: Record<LogLevelOrAll, number>;
}

/**
 * LogLevelFilter component for filtering log messages by severity level
 * Used in the Logs page to filter displayed log entries
 */
export function LogLevelFilter({ selectedLevel, onChange, counts = { debug: 0, info: 0, warning: 0, error: 0, all: 0 } }: LogLevelFilterProps) {
  const logLevels: Array<LogLevelOrAll> = ['all', LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR];
  
  // Icons for each log level
  const logLevelIcons = {
    [LogLevel.DEBUG]: <BsTerminal className="h-4 w-4" />,
    [LogLevel.INFO]: <BsInfoCircle className="h-4 w-4" />,
    [LogLevel.WARNING]: <BsExclamationTriangle className="h-4 w-4" />,
    [LogLevel.ERROR]: <BsExclamationCircle className="h-4 w-4" />,
    all: null
  };
  
  // Colors for each log level
  const logLevelColors = {
    [LogLevel.DEBUG]: 'bg-gray-200 text-gray-800',
    [LogLevel.INFO]: 'bg-blue-100 text-blue-800',
    [LogLevel.WARNING]: 'bg-yellow-100 text-yellow-800',
    [LogLevel.ERROR]: 'bg-red-100 text-red-800',
    all: 'bg-slate-100 text-slate-800'
  };

  return (
    <div className="flex flex-col space-y-2">
      <div className="text-sm font-medium">Log Level</div>
      <ToggleGroup 
        type="single" 
        value={selectedLevel} 
        onValueChange={(value) => onChange(value as LogLevelOrAll)}
        variant="outline"
        className="justify-start flex-wrap"
      >
        {logLevels.map((level) => (
          <ToggleGroupItem 
            key={level} 
            value={level}
            aria-label={`Filter by ${level} level`}
            className="flex items-center gap-2 capitalize"
            data-testid={`log-level-${level}`}
          >
            {logLevelIcons[level]}
            <span>{level}</span>
            {counts[level] > 0 && (
              <Badge 
                variant="secondary" 
                className={`text-xs ml-1 ${selectedLevel === level ? 'opacity-100' : 'opacity-70'}`}
              >
                {counts[level]}
              </Badge>
            )}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>

      <div className="flex mt-2 items-center justify-between pt-2">
        <div className="flex space-x-2">
          {/* Example log level badges for the legend */}
          {([LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR] as const).map(level => (
            <Badge 
              key={level} 
              variant="outline" 
              className={`${logLevelColors[level]} text-xs capitalize`}
            >
              {level}
            </Badge>
          ))}
        </div>
      </div>
    </div>
  );
}
