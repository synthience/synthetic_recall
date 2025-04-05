import React, { useState } from 'react';
import { useFeatures } from '../../contexts/FeaturesContext';
import { useMergeLog, useRuntimeConfig, useAssemblyLineage } from '../../lib/api';
import { ReconciledMergeLogEntry, LineageEntry } from '@shared/schema';

/**
 * Debug component for testing Phase 5.9 features
 * Displays feature availability and sample data from Phase 5.9 endpoints
 */
const Phase59Tester: React.FC = () => {
  const { explainabilityEnabled, isLoading: featuresLoading } = useFeatures();
  const [testAssemblyId, setTestAssemblyId] = useState<string>(''); 
  
  // Test merge log endpoint
  const { data: mergeLogData, isLoading: mergeLogLoading, isError: mergeLogError } = useMergeLog(5); // Limit to 5 entries
  
  // Test runtime config endpoint
  const { data: configData, isLoading: configLoading, isError: configError } = useRuntimeConfig('memory-core');
  
  // Test lineage endpoint (conditionally enabled when assembly ID is entered)
  const { 
    data: lineageData, 
    isLoading: lineageLoading, 
    isError: lineageError 
  } = useAssemblyLineage(testAssemblyId || null);

  if (featuresLoading) {
    return <div className="p-4">Loading features...</div>;
  }

  return (
    <div className="p-4 space-y-6 border rounded-lg">
      <h2 className="text-xl font-bold">Phase 5.9 Features Debug</h2>
      
      <div className="bg-gray-100 p-4 rounded-lg">
        <h3 className="text-lg font-semibold">Feature Flags</h3>
        <p className="py-2">
          Explainability Enabled: <span className={explainabilityEnabled ? 'text-green-600 font-bold' : 'text-red-600 font-bold'}>
            {explainabilityEnabled ? 'YES' : 'NO'}
          </span>
        </p>
        
        {!explainabilityEnabled && (
          <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4 my-2">
            <p>Explainability features are disabled in the Memory Core configuration.</p>
            <p>Enable them by setting <code>ENABLE_EXPLAINABILITY=true</code> in the Memory Core service.</p>
          </div>
        )}
      </div>

      {explainabilityEnabled && (
        <>
          {/* Runtime Config Test */}
          <div className="bg-gray-100 p-4 rounded-lg">
            <h3 className="text-lg font-semibold">Runtime Configuration</h3>
            {configLoading ? (
              <p>Loading configuration...</p>
            ) : configError ? (
              <p className="text-red-600">Error loading configuration</p>
            ) : (
              <pre className="bg-gray-800 text-green-400 p-4 rounded overflow-auto max-h-64">
                {JSON.stringify(configData, null, 2)}
              </pre>
            )}
          </div>
          
          {/* Merge Log Test */}
          <div className="bg-gray-100 p-4 rounded-lg">
            <h3 className="text-lg font-semibold">Merge Log</h3>
            {mergeLogLoading ? (
              <p>Loading merge log...</p>
            ) : mergeLogError ? (
              <p className="text-red-600">Error loading merge log</p>
            ) : !mergeLogData?.reconciled_log_entries?.length ? (
              <p>No merge log entries available</p>
            ) : (
              <div className="overflow-auto max-h-64">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Event ID</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {mergeLogData?.reconciled_log_entries?.map((entry: ReconciledMergeLogEntry, idx: number) => (
                      <tr key={idx}>
                        <td className="px-6 py-4 whitespace-nowrap">{new Date(entry.creation_timestamp).toLocaleString()}</td>
                        <td className="px-6 py-4 whitespace-nowrap">{entry.merge_event_id?.substring(0, 8)}...</td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                            entry.final_cleanup_status === 'completed' ? 'bg-green-100 text-green-800' : 
                            entry.final_cleanup_status === 'failed' ? 'bg-red-100 text-red-800' : 
                            'bg-yellow-100 text-yellow-800'
                          }`}>
                            {entry.final_cleanup_status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
          
          {/* Lineage Test */}
          <div className="bg-gray-100 p-4 rounded-lg">
            <h3 className="text-lg font-semibold">Assembly Lineage Test</h3>
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                value={testAssemblyId}
                onChange={(e) => setTestAssemblyId(e.target.value)}
                placeholder="Enter assembly ID"
                className="px-3 py-2 border rounded flex-grow"
              />
            </div>
            
            {testAssemblyId ? (
              lineageLoading ? (
                <p>Loading lineage...</p>
              ) : lineageError ? (
                <p className="text-red-600">Error loading lineage for assembly {testAssemblyId}</p>
              ) : !lineageData?.lineage?.length ? (
                <p>No lineage found for this assembly or assembly does not exist</p>
              ) : (
                <div className="overflow-auto max-h-64">
                  <h4 className="font-medium mb-2">Lineage Chain ({lineageData.lineage.length} entries):</h4>
                  <ul className="space-y-2">
                    {lineageData.lineage.map((entry: LineageEntry, idx: number) => (
                      <li key={idx} className="p-2 border rounded">
                        <div className="flex justify-between">
                          <span className="font-medium">{entry.assembly_id}</span>
                          <span className="text-sm text-gray-500">{entry.created_at ? new Date(entry.created_at).toLocaleString() : 'Unknown date'}</span>
                        </div>
                        <div className="text-sm">
                          <span className="text-gray-600">Status: </span>
                          {entry.status || 'Unknown'}
                        </div>
                        <div className="text-sm">
                          <span className="text-gray-600">Memories: </span>
                          {entry.memory_count || 'Unknown'}
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              )
            ) : (
              <p className="italic text-gray-500">Enter an assembly ID to test lineage lookup</p>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default Phase59Tester;
