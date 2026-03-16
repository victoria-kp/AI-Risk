"""pyrisk AI wrapper that uses the LangGraph pipeline.

Subclass of pyrisk's AI class. Implements reinforce(), attack(), and
freemove() by invoking the LangGraph player graph and returning the
structured actions pyrisk expects.
"""
