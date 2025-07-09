"""Enhanced streaming API route that supports both database results and live file streaming."""
import json
import time
import os
import sys
from pathlib import Path

# Add parent directory to path before imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, Response, request
from flask_cors import cross_origin
from db.simulation_results_utils import get_simulation_results
from db.simulation_catalog import SimulationCatalog
from pymongo import MongoClient

stream_live_bp = Blueprint('stream_live', __name__)


def find_stream_files(simulation_id):
    """Find streaming files for a simulation by checking file contents."""
    # Look in the main logs directory where stream files are created
    logs_dir = Path("logs")
    stream_files = []
    
    if logs_dir.exists():
        # Look for all stream files
        for stream_file in logs_dir.rglob("stream_*.jsonl"):
            try:
                # Check if this file belongs to our simulation
                with open(stream_file, 'r') as f:
                    first_line = f.readline()
                    if first_line and simulation_id in first_line:
                        stream_files.append(stream_file)
            except Exception:
                # Skip files we can't read
                pass
                
    return stream_files


def tail_file(file_path, from_line=0):
    """Read new lines from a file starting from a specific line."""
    with open(file_path, 'r') as f:
        # Skip to the starting line
        for _ in range(from_line):
            f.readline()
            
        # Read remaining lines
        lines = f.readlines()
        return lines, from_line + len(lines)


@stream_live_bp.route('/stream', methods=['GET'])
@cross_origin()
def stream_simulation():
    """
    Enhanced streaming endpoint that supports both:
    1. Completed simulations from database
    2. Live simulations from streaming files
    
    Query params:
    - id: simulation ID (required)
    - run_id: specific run ID (optional)
    - live: whether to stream from files (optional, default: true)
    """
    simulation_id = request.args.get('id')
    run_id = request.args.get('run_id')
    live_mode = request.args.get('live', 'true').lower() == 'true'
    
    if not simulation_id:
        return {'error': 'Simulation ID required'}, 400
    
    def generate():
        """Generator function for SSE streaming."""
        # Get total runs information if available
        total_runs = 0
        try:
            # First check if simulation is running
            client = MongoClient(os.environ.get("DB_CONNECTION_STRING", "mongodb://localhost:27017"))
            catalog_db = SimulationCatalog(client)
            sim_info = catalog_db.find_by_id(simulation_id)
            
            # Get expected runs from catalog
            if sim_info and 'expected_runs' in sim_info:
                total_runs = sim_info.get('expected_runs', 0)
            else:
                # Fall back to checking results
                results = get_simulation_results(simulation_id)
                if results and 'runs' in results:
                    total_runs = len(results.get('runs', []))
                elif results and 'messages' in results:
                    total_runs = 1
            
            is_running = sim_info and sim_info.get('status') in ['queued', 'running', 'started']
            
            # Send initial connection message with total runs info
            yield f"data: {json.dumps({'type': 'connected', 'simulation_id': simulation_id, 'total_runs': total_runs})}\n\n"
            
            # If simulation is running and live mode is enabled, try file streaming
            if is_running and live_mode:
                yield f"data: {json.dumps({'type': 'status', 'status': 'live_streaming', 'mode': 'file', 'total_runs': total_runs})}\n\n"
                
                # Find stream files by checking their contents for the simulation_id
                stream_files = find_stream_files(simulation_id)
                
                if stream_files:
                    # Use the most recent stream file
                    stream_file = max(stream_files, key=lambda f: f.stat().st_mtime)
                    yield f"data: {json.dumps({'type': 'debug', 'message': f'Reading from {stream_file}'})}\n\n"
                    
                    # Stream from file
                    last_line = 0
                    empty_reads = 0
                    max_empty_reads = 30  # Stop after 30 seconds of no new data
                    
                    while empty_reads < max_empty_reads:
                        try:
                            lines, last_line = tail_file(stream_file, last_line)
                            
                            if lines:
                                empty_reads = 0
                                for line in lines:
                                    if line.strip():
                                        try:
                                            event = json.loads(line)
                                            # Convert file event to streaming format
                                            if event.get('type') == 'message':
                                                # Get run_id from metadata if available
                                                event_run_id = event.get('metadata', {}).get('run_id') or event.get('run_id') or run_id or 'default'
                                                yield f"data: {json.dumps({'type': 'message', 'run_id': event_run_id, 'agent': event.get('agent'), 'content': event.get('content'), 'timestamp': event.get('timestamp', time.time())})}\n\n"
                                            elif event.get('type') == 'complete':
                                                yield f"data: {json.dumps({'type': 'complete', 'status': 'finished'})}\n\n"
                                                return
                                        except json.JSONDecodeError:
                                            pass
                            else:
                                empty_reads += 1
                                time.sleep(1)  # Wait for new data
                                
                        except FileNotFoundError:
                            # File might not exist yet
                            time.sleep(1)
                            empty_reads += 1
                    
                    # If we exit the loop, simulation might still be running but no new data
                    yield f"data: {json.dumps({'type': 'status', 'status': 'waiting', 'message': 'No new messages'})}\n\n"
            
            # Fall back to database results
            results = get_simulation_results(simulation_id)
            
            if results:
                yield f"data: {json.dumps({'type': 'status', 'status': 'streaming', 'mode': 'database', 'total_runs': total_runs})}\n\n"
                
                # Check if results has 'messages' directly (single run)
                if 'messages' in results:
                    messages = results.get('messages', [])
                    for idx, msg in enumerate(messages):
                        yield f"data: {json.dumps({'type': 'message', 'run_id': run_id or 'default', 'agent': msg.get('agent'), 'content': msg.get('message'), 'timestamp': time.time() + idx * 0.1})}\n\n"
                        time.sleep(0.05)  # Small delay for visual effect
                        
                # Also check for 'runs' structure (multiple runs)
                elif results.get('runs'):
                    for run_idx, run in enumerate(results.get('runs', [])):
                        run_id_value = run.get('run_id', f'run_{run_idx}')
                        
                        # Skip if specific run_id requested and doesn't match
                        if run_id and run_id_value != run_id:
                            continue
                        
                        # Send all messages from this run
                        messages = run.get('messages', [])
                        for idx, msg in enumerate(messages):
                            yield f"data: {json.dumps({'type': 'message', 'run_id': run_id_value, 'agent': msg.get('agent'), 'content': msg.get('message'), 'timestamp': time.time() + idx * 0.1})}\n\n"
                            time.sleep(0.05)  # Small delay for visual effect
            else:
                # No results yet
                yield f"data: {json.dumps({'type': 'status', 'status': 'no_data', 'message': 'Simulation has no results yet'})}\n\n"
            
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
            'X-Accel-Buffering': 'no',  # Disable Nginx buffering
            'Connection': 'keep-alive'
        }
    )


@stream_live_bp.route('/stream/status', methods=['GET'])
@cross_origin()
def stream_status():
    """Get the current status of a simulation without streaming."""
    simulation_id = request.args.get('id')
    
    if not simulation_id:
        return {'error': 'Simulation ID required'}, 400
    
    # Check catalog for simulation status
    client = MongoClient(os.environ.get("DB_CONNECTION_STRING", "mongodb://localhost:27017"))
    catalog_db = SimulationCatalog(client)
    sim_info = catalog_db.find_by_id(simulation_id)
    
    if not sim_info:
        return {'status': 'not_found'}, 404
    
    status_info = {
        'status': sim_info.get('status', 'unknown'),
        'name': sim_info.get('name', ''),
        'created_at': sim_info.get('created_at', ''),
        'completed_runs': sim_info.get('progress_percentage', 0),
        'total_runs': sim_info.get('expected_runs', 0)
    }
    
    # Check for stream files
    stream_files = find_stream_files(simulation_id)
    status_info['has_stream_files'] = len(stream_files) > 0
    
    # Check results
    results = get_simulation_results(simulation_id)
    if results:
        # Handle direct messages structure
        if 'messages' in results:
            status_info.update({
                'has_results': True,
                'runs': 1,
                'has_messages': len(results.get('messages', [])) > 0
            })
        # Handle runs structure
        elif 'runs' in results:
            status_info.update({
                'has_results': True,
                'runs': len(results.get('runs', [])),
                'has_messages': any(run.get('messages') for run in results.get('runs', []))
            })
        else:
            status_info.update({
                'has_results': True,
                'runs': 0,
                'has_messages': False
            })
    else:
        status_info['has_results'] = False
    
    return status_info