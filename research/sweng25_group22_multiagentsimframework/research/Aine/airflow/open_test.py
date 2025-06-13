from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.exporter import OTLPSpanExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Create a TracerProvider
trace.set_tracer_provider(TracerProvider())

RequestsInstrumentor().instrument()

# Add exporters (Console Exporter for development, OTLP for production)
span_exporter = OTLPSpanExporter(endpoint="your-otel-collector-endpoint", insecure=True)
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(span_exporter))

# Alternatively, you can use the console exporter for debugging purposes
trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# Now the app will start tracing
tracer = trace.get_tracer(__name__)

# Example of starting a span (trace)
with tracer.start_as_current_span("foo"):
    print("Tracing!")
