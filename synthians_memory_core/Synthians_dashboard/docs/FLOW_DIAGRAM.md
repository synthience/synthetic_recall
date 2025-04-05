
--

7. Flow Diagram Example (Mermaid in docs/dashboard/FLOW_DIAGRAM.md)
# Dashboard Data Flow Diagrams

## Example: Explaining Assembly Merge

```mermaid
sequenceDiagram
    participant User
    participant FE_Component as AssemblyDetail.tsx
    participant FE_Hook as useExplainMerge (api.ts)
    participant FE_Proxy as Dashboard Backend (routes.ts)
    participant BE_Service as Memory Core API

    User->>+FE_Component: Clicks "Explain Merge" Button
    FE_Component->>+FE_Hook: Calls explainMergeQuery.refetch()
    FE_Hook->>+FE_Proxy: GET /api/memory-core/assemblies/{id}/explain_merge
    Note over FE_Proxy: TODO: Implement this Proxy Route
    FE_Proxy->>+BE_Service: GET {MEMORY_CORE_URL}/assemblies/{id}/explain_merge
    BE_Service-->>-FE_Proxy: JSON Response (Merge Data or Error)
    FE_Proxy-->>-FE_Hook: Forward JSON Response
    FE_Hook-->>-FE_Component: TanStack Query updates data/error state
    FE_Component->>User: Renders MergeExplanationView with data or error
Use code with caution.
Markdown
Example: Loading Merge Log
sequenceDiagram
    participant User
    participant FE_Component as MergeLogPage.tsx (or similar)
    participant FE_Hook as useMergeLog (api.ts)
    participant FE_Proxy as Dashboard Backend (routes.ts)
    participant BE_Service as Memory Core API

    User->>+FE_Component: Navigates to Log Page
    FE_Component->>+FE_Hook: Renders component using useMergeLog(limit)
    Note over FE_Hook: TanStack Query automatically fetches on mount
    FE_Hook->>+FE_Proxy: GET /api/memory-core/diagnostics/merge_log?limit=50
    Note over FE_Proxy: TODO: Implement this Proxy Route
    FE_Proxy->>+BE_Service: GET {MEMORY_CORE_URL}/diagnostics/merge_log?limit=50
    BE_Service-->>-FE_Proxy: JSON Response (Log Entries or Error)
    FE_Proxy-->>-FE_Hook: Forward JSON Response
    FE_Hook-->>-FE_Component: TanStack Query provides data/state
    FE_Component->>User: Renders MergeLogView with data
Use code with caution.
Mermaid
(Add similar diagrams for other key interactions like fetching stats, config, lineage, activation explanation)

---

This documentation suite provides a solid foundation for understanding the dashboard project and tackling the Phase 5.9 integration work. Remember to update the **TODO** sections in the actual code (`server/routes.ts`, `client/src/lib/api.ts`, `shared/schema.ts`) as you implement the necessary connections.