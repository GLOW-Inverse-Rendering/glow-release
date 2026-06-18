import mitsuba as mi
from nerad.sensors import register_sensor

@register_sensor("random_ray")
class RandomRay(mi.Sensor):
    def __init__(self, props):
        # print(props)
        # exit()
        super().__init__(props)
        self.wrapped_sensor = props["wrapped_sensor"]
        
    def sample_ray(time, sample1, sample2, sample3, active):
        
        pass
    
    def sample_ray_differential(time, sample1, sample2, sample3, active):
        pass

    def sample_direction(ref, sample, active):
        pass

    # def pdf_direction(ref, ds, active):
    #     pass

    # def eval_direction(ref, ds, active):
    #     pass

    # def sample_position(time, sample, active):
    #     pass

    # def pdf_position(ps, active):
    #     pass

    # def eval(si, active):
    #     pass

    # def sample_wavelengths(si, sample, active):
    #     pass

    def bbox():
        return self.wrapped_sensor.bbox()

    def traverse(callback):
        self.wrapped_sensor.traverse(callback)
        super().traversal(callback)
        pass

    def parameters_changed(keys):
        self.wrapped_sensor.parameters_changed(callback)
        super().parameters_changed(keys)
        pass
    
        
    
