import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { Badge } from '@/components/ui/badge';
import { BsLayers, BsCpu, BsLightningCharge } from 'react-icons/bs';

type ServiceType = 'memory-core' | 'neural-memory' | 'cce' | 'all';

interface ServiceFilterProps {
  selectedService: ServiceType;
  onChange: (service: ServiceType) => void;
  counts?: Partial<Record<ServiceType, number>>;
}

/**
 * ServiceFilter component for filtering log messages by service
 * Used in the Logs page to filter displayed log entries by service source
 */
export function ServiceFilter({ selectedService, onChange, counts = { 'all': 0, 'memory-core': 0, 'neural-memory': 0, 'cce': 0 } }: ServiceFilterProps) {
  const services: ServiceType[] = ['all', 'memory-core', 'neural-memory', 'cce'];
  
  // Display names for services
  const serviceNames = {
    'all': 'All Services',
    'memory-core': 'Memory Core',
    'neural-memory': 'Neural Memory',
    'cce': 'Context Cascade'
  };
  
  // Icons for each service
  const serviceIcons = {
    'memory-core': <BsLayers className="h-4 w-4" />,
    'neural-memory': <BsCpu className="h-4 w-4" />,
    'cce': <BsLightningCharge className="h-4 w-4" />,
    'all': null
  };

  return (
    <div className="flex flex-col space-y-2">
      <div className="text-sm font-medium">Service</div>
      <ToggleGroup 
        type="single" 
        value={selectedService} 
        onValueChange={(value) => onChange(value as ServiceType)}
        variant="outline"
        className="justify-start flex-wrap"
      >
        {services.map((service) => (
          <ToggleGroupItem 
            key={service} 
            value={service}
            aria-label={`Filter by ${serviceNames[service]}`}
            className="flex items-center gap-2"
            data-testid={`service-${service}`}
          >
            {serviceIcons[service]}
            <span className="whitespace-nowrap">{serviceNames[service]}</span>
            {counts[service] && counts[service]! > 0 && (
              <Badge 
                variant="secondary" 
                className={`text-xs ml-1 ${selectedService === service ? 'opacity-100' : 'opacity-70'}`}
              >
                {counts[service]}
              </Badge>
            )}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
    </div>
  );
}
