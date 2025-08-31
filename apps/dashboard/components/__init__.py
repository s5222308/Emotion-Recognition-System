"""
Component registry for dashboard
This module manages all dashboard components and their registration.
"""

class ComponentRegistry:
    """Registry for dashboard components"""
    
    def __init__(self):
        self.components = {}
    
    def register_component(self, name, component_class):
        """Register a dashboard component"""
        self.components[name] = component_class
    
    def get_component(self, name):
        """Get a registered component"""
        return self.components.get(name)
    
    def get_all_components(self):
        """Get all registered components"""
        return self.components

# Global registry instance
registry = ComponentRegistry()

def register_component(name):
    """Decorator for registering components"""
    def decorator(component_class):
        registry.register_component(name, component_class)
        return component_class
    return decorator

__all__ = ['ComponentRegistry', 'registry', 'register_component']