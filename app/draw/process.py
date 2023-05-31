import time
from multiprocessing import Process

import pyglet
from UltraDict import UltraDict


class SharedContainer():
    def __init__(self, memory_name=None):
        self.shared_dict = UltraDict({}, name=memory_name)
        self.shared_memory_name = self.shared_dict.name

    def set_mlp(self, mlp):
        self.shared_dict['mlp'] = mlp

    def get_mlp(self):
        return self.shared_dict['mlp']

    def set_loss(self, loss):
        self.shared_dict['loss'] = loss

    def get_loss(self):
        return self.shared_dict['loss']

    def set_parameters(self, parameters):
        self.shared_dict['parameters'] = parameters
        self.shared_dict['updated'] = time.time()

    def get_parameters(self):
        return self.shared_dict['parameters']

    def set_parameters_map(self, parameters):
        self.shared_dict['parameters_map'] = parameters

    def get_parameters_map(self):
        return self.shared_dict['parameters_map']

    def get_parameter(self, id):
        return self.shared_dict[id]

    def set_result(self, result):
        self.shared_dict['result'] = result

    def get_result(self):
        return self.shared_dict['result']

    def mark_as_updated(self):
        self.shared_dict['refreshed'] = time.time()

    def should_update(self):
        updated = self.shared_dict.get("updated")
        refreshed = self.shared_dict.get("refreshed")
        update_detected = False
        if updated and not refreshed:
            print("detected update1")
            update_detected = True
        if updated and refreshed and updated > refreshed:
            print("detected update2")
            update_detected = True
        return update_detected


class PygletProcess():
    def __init__(self):
        self.jobs = []
        self.manager = None
        self.update_ui_func = None
        self.on_job_finished = None

    def is_active(self):
        return len(self.jobs) > 0

    def kill(self):
        if len(self.jobs) == 0: return
        for j in self.jobs: j.terminate()
        self.jobs.clear()
        print("Job killed")
        pyglet.clock.unschedule(self.monitor_the_job)
        pyglet.clock.schedule_once(self.on_finished, delay=0)

    def on_finished(self, dt):
        if self.on_job_finished:
            self.on_job_finished()

    def monitor_the_job(self, dt, job):
        if job.is_alive():
            if self.update_ui_func:
                self.update_ui_func()
            return
        self.kill()

    def launch(self,
               target,
               args,
               update_ui_func=None,
               on_job_finished=None):
        self.update_ui_func = None
        self.on_job_finished = None
        self.kill()
        self.update_ui_func = update_ui_func
        self.on_job_finished = on_job_finished
        p = Process(target=target, args=args)
        self.jobs.append(p)
        p.start()
        pyglet.clock.schedule_interval(self.monitor_the_job, 1.0, p)
