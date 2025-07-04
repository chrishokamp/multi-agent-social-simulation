"""Clean API route for streaming simulation conversations."""
import json
import time
from flask import Blueprint, Response, request
from flask_cors import cross_origin
from db.simulation_results_utils import get_simulation_results

stream_bp = Blueprint('stream', __name__)


@stream_bp.route('/stream', methods=['GET'])
@cross_origin()
def stream_simulation():
    """
    Stream simulation messages as they're generated.
    
    Query params:
    - id: simulation ID (required)
    - run_id: specific run ID (optional)
    """
    simulation_id = request.args.get('id')
    run_id = request.args.get('run_id')
    
    if not simulation_id:
        return {'error': 'Simulation ID required'}, 400
    
    def generate():
        """Generator function for SSE streaming."""
        # Send initial connection message
        yield f"data: {json.dumps({'type': 'connected', 'simulation_id': simulation_id})}\n\n"
        
        try:
            # Get results from database
            results = get_simulation_results(simulation_id)
            
            if results and results.get('runs'):
                # Process all messages from all runs
                for run_idx, run in enumerate(results.get('runs', [])):
                    run_id_value = run.get('run_id', f'run_{run_idx}')
                    
                    # Skip if specific run_id requested and doesn't match
                    if run_id and run_id_value != run_id:
                        continue
                    
                    # Send all messages from this run
                    messages = run.get('messages', [])
                    for idx, msg in enumerate(messages):
                        yield f"data: {json.dumps({
                            'type': 'message',
                            'run_id': run_id_value,
                            'agent': msg.get('agent'),
                            'content': msg.get('message'),
                            'timestamp': time.time() + idx * 0.1
                        })}\n\n"
            
            # Send completion message
            yield f"data: {json.dumps({'type': 'complete', 'status': 'finished'})}\n\n"
            
        except Exception as e:
            # Send error message
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'  # Disable Nginx buffering
        }
    )


@stream_bp.route('/stream/status', methods=['GET'])
@cross_origin()
def stream_status():
    """Get the current status of a simulation without streaming."""
    simulation_id = request.args.get('id')
    
    if not simulation_id:
        return {'error': 'Simulation ID required'}, 400
    
    # Check results
    results = get_simulation_results(simulation_id)
    if results:
        return {
            'status': 'complete',
            'runs': len(results.get('runs', [])),
            'has_messages': any(run.get('messages') for run in results.get('runs', []))
        }
    
    return {'status': 'not_found'}, 404