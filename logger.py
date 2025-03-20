import logging

# Configure loggers
def setup_loggers():
    # Main logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(ch)
    
    return logger

# Create specialized loggers
debug_logger = logging.getLogger('debug')
rerank_logger = logging.getLogger('rerank')
embed_logger = logging.getLogger('embed')

# Set log levels
debug_logger.setLevel(logging.INFO)
rerank_logger.setLevel(logging.INFO)
embed_logger.setLevel(logging.INFO)