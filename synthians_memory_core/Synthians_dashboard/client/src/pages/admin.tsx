import React, { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { verifyMemoryCoreIndex, triggerMemoryCoreRetryLoop, initializeNeuralMemory, setCCEVariant } from "@/lib/api";

export default function Admin() {
  const { toast } = useToast();
  const [selectedVariant, setSelectedVariant] = useState("MAC");
  const [isLoading, setIsLoading] = useState({
    verifyIndex: false,
    retryLoop: false,
    initNM: false,
    setVariant: false
  });
  const [lastActionResult, setLastActionResult] = useState<{
    action: string;
    success: boolean;
    message: string;
  } | null>(null);
  
  // Handle verify index action
  const handleVerifyIndex = async () => {
    setIsLoading({ ...isLoading, verifyIndex: true });
    try {
      await verifyMemoryCoreIndex();
      toast({
        title: "Success",
        description: "Index verification triggered successfully",
      });
      setLastActionResult({
        action: "Verify Memory Core Index",
        success: true,
        message: "Index verification has been triggered. This process will run in the background."
      });
    } catch (error) {
      console.error("Failed to verify index:", error);
      toast({
        title: "Error",
        description: "Failed to trigger index verification",
        variant: "destructive"
      });
      setLastActionResult({
        action: "Verify Memory Core Index",
        success: false,
        message: `Error: ${(error as Error).message || "Unknown error occurred"}`
      });
    } finally {
      setIsLoading({ ...isLoading, verifyIndex: false });
    }
  };
  
  // Handle retry loop action
  const handleRetryLoop = async () => {
    setIsLoading({ ...isLoading, retryLoop: true });
    try {
      await triggerMemoryCoreRetryLoop();
      toast({
        title: "Success",
        description: "Retry loop triggered successfully",
      });
      setLastActionResult({
        action: "Trigger Retry Loop",
        success: true,
        message: "Retry loop has been triggered. Pending operations will be reprocessed."
      });
    } catch (error) {
      console.error("Failed to trigger retry loop:", error);
      toast({
        title: "Error",
        description: "Failed to trigger retry loop",
        variant: "destructive"
      });
      setLastActionResult({
        action: "Trigger Retry Loop",
        success: false,
        message: `Error: ${(error as Error).message || "Unknown error occurred"}`
      });
    } finally {
      setIsLoading({ ...isLoading, retryLoop: false });
    }
  };
  
  // Handle Neural Memory initialization
  const handleInitializeNM = async () => {
    setIsLoading({ ...isLoading, initNM: true });
    try {
      await initializeNeuralMemory();
      toast({
        title: "Success",
        description: "Neural Memory initialized successfully",
      });
      setLastActionResult({
        action: "Initialize Neural Memory",
        success: true,
        message: "Neural Memory module has been reinitialized."
      });
    } catch (error) {
      console.error("Failed to initialize Neural Memory:", error);
      toast({
        title: "Error",
        description: "Failed to initialize Neural Memory",
        variant: "destructive"
      });
      setLastActionResult({
        action: "Initialize Neural Memory",
        success: false,
        message: `Error: ${(error as Error).message || "Unknown error occurred"}`
      });
    } finally {
      setIsLoading({ ...isLoading, initNM: false });
    }
  };
  
  // Handle CCE variant selection
  const handleSetVariant = async () => {
    setIsLoading({ ...isLoading, setVariant: true });
    try {
      await setCCEVariant(selectedVariant);
      toast({
        title: "Success",
        description: `Variant set to ${selectedVariant} successfully`,
      });
      setLastActionResult({
        action: "Set CCE Variant",
        success: true,
        message: `CCE Variant has been set to ${selectedVariant}. This will affect future responses.`
      });
    } catch (error) {
      console.error("Failed to set CCE variant:", error);
      toast({
        title: "Error",
        description: "Failed to set CCE variant",
        variant: "destructive"
      });
      setLastActionResult({
        action: "Set CCE Variant",
        success: false,
        message: `Error: ${(error as Error).message || "Unknown error occurred"}`
      });
    } finally {
      setIsLoading({ ...isLoading, setVariant: false });
    }
  };

  return (
    <>
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-white mb-1">Admin Actions</h2>
        <p className="text-sm text-gray-400">
          Manually trigger maintenance tasks for testing and debugging
        </p>
      </div>
      
      <Alert className="mb-6 border-yellow-600 bg-yellow-950/30">
        <i className="fas fa-exclamation-triangle text-yellow-400 mr-2"></i>
        <AlertTitle>Warning: Administrative Area</AlertTitle>
        <AlertDescription>
          These actions can affect the performance and behavior of the Synthians Cognitive Architecture services.
          Use with caution in production environments.
        </AlertDescription>
      </Alert>
      
      {lastActionResult && (
        <Alert 
          className={`mb-6 ${lastActionResult.success 
            ? "border-green-600 bg-green-950/30" 
            : "border-red-600 bg-red-950/30"}`}
        >
          <i className={`fas ${lastActionResult.success ? "fa-check-circle text-green-400" : "fa-times-circle text-red-400"} mr-2`}></i>
          <AlertTitle>{lastActionResult.action} - {lastActionResult.success ? "Success" : "Failed"}</AlertTitle>
          <AlertDescription>
            {lastActionResult.message}
          </AlertDescription>
        </Alert>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Memory Core Actions */}
        <Card>
          <CardHeader>
            <CardTitle>Memory Core Actions</CardTitle>
            <CardDescription>Maintenance operations for the Memory Core service</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h3 className="text-sm font-medium mb-2">Verify Vector Index</h3>
              <p className="text-sm text-gray-400 mb-4">
                Triggers a background job to verify the integrity of the vector index.
                This will compare indexed vectors against their source memories.
              </p>
              <Button 
                onClick={handleVerifyIndex} 
                disabled={isLoading.verifyIndex}
                className="w-full"
              >
                {isLoading.verifyIndex && <i className="fas fa-spin fa-spinner mr-2"></i>}
                Verify Index
              </Button>
            </div>
            
            <Separator />
            
            <div>
              <h3 className="text-sm font-medium mb-2">Trigger Retry Loop</h3>
              <p className="text-sm text-gray-400 mb-4">
                Forces the Memory Core to retry any pending or failed operations,
                such as vector updates or assembly indexing.
              </p>
              <Button 
                onClick={handleRetryLoop} 
                disabled={isLoading.retryLoop}
                className="w-full"
              >
                {isLoading.retryLoop && <i className="fas fa-spin fa-spinner mr-2"></i>}
                Trigger Retry Loop
              </Button>
            </div>
          </CardContent>
        </Card>
        
        {/* Neural Memory Actions */}
        <Card>
          <CardHeader>
            <CardTitle>Neural Memory Actions</CardTitle>
            <CardDescription>Operations for the Neural Memory module</CardDescription>
          </CardHeader>
          <CardContent>
            <div>
              <h3 className="text-sm font-medium mb-2">Initialize Neural Memory</h3>
              <p className="text-sm text-gray-400 mb-4">
                Reinitializes the Neural Memory module, resetting its internal state.
                This is useful if the module becomes unstable or unresponsive.
              </p>
              <Button 
                onClick={handleInitializeNM} 
                disabled={isLoading.initNM}
                variant="destructive"
                className="w-full"
              >
                {isLoading.initNM && <i className="fas fa-spin fa-spinner mr-2"></i>}
                Reset Neural Memory
              </Button>
              <p className="text-xs text-destructive mt-2">
                <i className="fas fa-exclamation-circle mr-1"></i>
                Warning: This will reset any in-progress emotional loop training.
              </p>
            </div>
          </CardContent>
        </Card>
        
        {/* CCE Actions */}
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Context Cascade Engine Actions</CardTitle>
            <CardDescription>Control operations for the CCE service</CardDescription>
          </CardHeader>
          <CardContent>
            <div>
              <h3 className="text-sm font-medium mb-2">Set CCE Variant</h3>
              <p className="text-sm text-gray-400 mb-4">
                Manually override the active variant used by the Context Cascade Engine.
                This will bypass the automatic selection mechanism.
              </p>
              <div className="flex space-x-4">
                <Select value={selectedVariant} onValueChange={setSelectedVariant}>
                  <SelectTrigger className="w-[180px]">
                    <SelectValue placeholder="Select variant" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="MAC">MAC</SelectItem>
                    <SelectItem value="MAG">MAG</SelectItem>
                    <SelectItem value="MAL">MAL</SelectItem>
                  </SelectContent>
                </Select>
                
                <Button 
                  onClick={handleSetVariant} 
                  disabled={isLoading.setVariant}
                  className="flex-1"
                >
                  {isLoading.setVariant && <i className="fas fa-spin fa-spinner mr-2"></i>}
                  Set Variant to {selectedVariant}
                </Button>
              </div>
            </div>
          </CardContent>
          <CardFooter className="bg-muted/50 flex justify-between">
            <p className="text-xs text-gray-400">
              <i className="fas fa-info-circle mr-1"></i>
              These endpoints may respond with a 501 Not Implemented if the backend service does not support them yet.
            </p>
            
            <Button variant="ghost" size="sm" onClick={() => setLastActionResult(null)}>
              Clear Results
            </Button>
          </CardFooter>
        </Card>
      </div>
    </>
  );
}
