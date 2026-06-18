import mitsuba as mi
import drjit as dr


def put_flashlight_next_to_camera(sensor, params, type, offset=None):
    transform = sensor.world_transform()
    if type == "flashlight":
        position = dr.zeros(mi.Point3f)
        if offset is not None:
            print("found global offset", offset)
            position += offset
        position = transform @ position    
        params['flashlight.position'] = position
    elif type == "spot":
        assert offset is None
        params['flashlight.to_world'] = transform
        pass
    else:
        raise NotImplementedError(type)

    params.update()
