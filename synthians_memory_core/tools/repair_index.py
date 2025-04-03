# Inside api/server.py
@app.post("/repair_index")
async def repair_index_endpoint(request: Request): # Make async
    try:
        body = await request.json()
        repair_type = body.get("repair_type", "auto")
        logger.info(f"Repair index request received with repair_type: {repair_type}")

        if not hasattr(app.state, 'memory_core') or app.state.memory_core is None:
             raise HTTPException(status_code=500, detail="Memory core not initialized")

        # Call the core's repair method
        result = await app.state.memory_core.repair_index(repair_type)
        status_code = 200 if result.get("success") else 500
        return JSONResponse(content=result, status_code=status_code)

    except json.JSONDecodeError:
         raise HTTPException(status_code=400, detail="Invalid JSON body")
    except Exception as e:
        logger.error(f"Error repairing vector index: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")