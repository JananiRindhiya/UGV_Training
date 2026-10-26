import pygame, math, random, csv, time
from pathlib import Path
from dataclasses import dataclass

W, H = 1280, 760
MAP = 60.0
FPS = 60

@dataclass
class Obstacle:
    x: float; y: float; w: float; h: float; kind: str = 'ROCK'

class Camera:
    def __init__(self):
        self.yaw = 35.0
        self.pitch = 48.0
        self.distance = 34.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.follow = True
        self.drag_left = False
        self.drag_right = False
        self.last_mouse = (0, 0)

    def begin_drag(self, button, pos):
        if button == 1: self.drag_left = True
        if button == 3: self.drag_right = True
        self.last_mouse = pos

    def end_drag(self, button):
        if button == 1: self.drag_left = False
        if button == 3: self.drag_right = False

    def mouse_motion(self, pos):
        dx = pos[0] - self.last_mouse[0]; dy = pos[1] - self.last_mouse[1]
        self.last_mouse = pos
        if self.drag_left:
            self.yaw += dx * 0.45
            self.pitch = max(22, min(75, self.pitch - dy * 0.35))
            self.follow = False
        if self.drag_right:
            r = math.radians(self.yaw)
            scale = self.distance * 0.008
            self.target_x += (-math.cos(r)*dx + math.sin(r)*dy) * scale
            self.target_y += (-math.sin(r)*dx - math.cos(r)*dy) * scale
            self.follow = False

    def zoom(self, amount):
        self.distance = max(8, min(90, self.distance * (0.88 ** amount)))

    def reset(self):
        self.yaw, self.pitch, self.distance = 35, 48, 34
        self.target_x = self.target_y = 0
        self.follow = True

    def update(self, ugv):
        # UGV state is stored as a dictionary.
        if self.follow:
            self.target_x += (ugv['x'] - self.target_x) * 0.08
            self.target_y += (ugv['y'] - self.target_y) * 0.08

    def project(self, x, y, z=0):
        # Orbit camera: horizontal rotation, then perspective-like vertical projection.
        dx, dy = x-self.target_x, y-self.target_y
        a = math.radians(-self.yaw)
        cx = dx*math.cos(a) - dy*math.sin(a)
        cy = dx*math.sin(a) + dy*math.cos(a)
        elev = math.radians(self.pitch)
        depth = cy*math.cos(elev) + z*math.sin(elev)
        vertical = cy*math.sin(elev) - z*math.cos(elev)
        scale = 520 / max(8, self.distance - depth*0.30)
        sx = W/2 + cx*scale
        sy = H*0.52 + vertical*scale
        return sx, sy, scale, depth


class AdaptiveAI:
    """Lightweight adaptive training policy for the desktop simulator.

    It uses recent training outcomes to estimate skill and recommend the next
    difficulty. This is intentionally transparent so the term paper can show
    how the adaptation decision is made.
    """
    def __init__(self):
        self.history = []
        self.recommended = 2
        self.last_reason = 'Collecting baseline performance'

    def score_run(self, success, elapsed, collisions, difficulty):
        # Success is the strongest signal; collisions and time refine the score.
        score = 100.0 if success else 20.0
        score -= min(35.0, collisions * 5.0)
        if success:
            score += max(0.0, 15.0 - max(0.0, elapsed - 25.0) * 0.20)
        else:
            score -= min(15.0, max(0.0, elapsed - 60.0) * 0.15)
        score = max(0.0, min(100.0, score))
        self.history.append({
            'success': bool(success), 'elapsed': float(elapsed),
            'collisions': int(collisions), 'difficulty': int(difficulty),
            'score': round(score, 1)
        })
        self.history = self.history[-5:]
        avg = sum(x['score'] for x in self.history) / len(self.history)
        success_rate = sum(x['success'] for x in self.history) / len(self.history)
        if len(self.history) >= 2 and avg >= 78 and success_rate >= 0.60:
            self.recommended = min(3, difficulty + 1)
            self.last_reason = f'Strong performance (score {avg:.0f}) → increase challenge'
        elif avg <= 48 or (len(self.history) >= 2 and success_rate <= 0.25):
            self.recommended = max(1, difficulty - 1)
            self.last_reason = f'Performance needs support (score {avg:.0f}) → reduce challenge'
        else:
            self.recommended = difficulty
            self.last_reason = f'Performance stable (score {avg:.0f}) → keep challenge'
        return score

    def skill_score(self):
        if not self.history:
            return 50.0
        return sum(x['score'] for x in self.history) / len(self.history)

    def recommendation_text(self):
        return f'Next difficulty: {self.recommended} | {self.last_reason}'

class Simulator:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('UGV Adaptive Training Simulator - Desktop 3D Camera')
        self.screen = pygame.display.set_mode((W,H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('segoeui', 24, bold=True)
        self.small = pygame.font.SysFont('segoeui', 17)
        self.big = pygame.font.SysFont('segoeui', 32, bold=True)
        self.title = pygame.font.SysFont('segoeui', 38, bold=True)
        self.camera = Camera()
        self.ai = AdaptiveAI()
        self.adaptive_mode = True
        self.current_run_logged = False
        self.difficulty = 2
        self.mode = 'MANUAL'
        self.seed = 1
        self.log_path = Path(__file__).resolve().parent / 'training_results.csv'
        self.run_start = time.time()
        self.max_run_time = 90.0
        self.dashboard = False
        self.dashboard_scroll = 0
        self.all_results = self.load_results()
        self.restore_ai_history()
        self.reset_run(log_previous=False)

    def load_results(self):
        rows=[]
        if self.log_path.exists():
            try:
                with self.log_path.open(newline='', encoding='utf-8') as f:
                    rows=list(csv.DictReader(f))
            except Exception:
                rows=[]
        return rows[-30:]

    def restore_ai_history(self):
        self.ai.history = []
        for row in self.all_results[-5:]:
            try:
                self.ai.history.append({
                    'success': bool(int(float(row.get('success', 0)))),
                    'elapsed': float(row.get('time_sec', 0)),
                    'collisions': int(float(row.get('collisions', 0))),
                    'hazard_hits': int(float(row.get('hazard_hits', 0))),
                    'difficulty': int(float(row.get('difficulty', 2))),
                    'score': float(row.get('score', 50))
                })
            except (ValueError, TypeError):
                continue
        if self.ai.history:
            avg = sum(x['score'] for x in self.ai.history) / len(self.ai.history)
            self.ai.recommended = 3 if avg >= 78 else 1 if avg <= 48 else self.difficulty
            self.ai.last_reason = 'Loaded previous training history from CSV'

    def refresh_results(self):
        self.all_results = self.load_results()
        self.restore_ai_history()

    def reset_run(self, log_previous=True):
        # If the user resets after actually driving, keep that attempt in the
        # analytics instead of silently deleting it. This makes every training
        # attempt measurable, even when the goal was not reached.
        if log_previous and hasattr(self, 'run_start') and hasattr(self, 'ugv') and not self.current_run_logged:
            elapsed = max(0.0, time.time() - self.run_start)
            moved = math.hypot(self.ugv['x'] + 22.0, self.ugv['y'] + 22.0) > 0.5
            if elapsed > 2.0 or moved or self.collisions > 0:
                self.finish_run(False, 'RUN RESET — ATTEMPT LOGGED')
        self.ugv = {'x':-22.0,'y':-22.0,'heading':45.0,'speed':0.0}
        self.goal = (22.0,22.0)
        self.collisions = 0
        self.hazard_hits = 0
        self.hazard_hit_types = {'ROCK': 0, 'POTHOLE': 0, 'LOG': 0, 'BOULDER': 0}
        self.stuck = False
        self.success = False
        self.status = 'READY'
        self.elapsed = 0
        self.lidar = []
        self.replan_timer = 0
        self.escape_timer = 0
        self.escape_dir = 1
        self.path = []
        self.generate_terrain()
        if not hasattr(self, 'camera'):
            self.camera = Camera()
        self.camera.reset()
        self.run_start = time.time()
        self.current_run_logged = False
        self.run_score = None

    def generate_terrain(self):
        rng = random.Random(self.seed)
        # More variety as difficulty increases. Rocks block the UGV; potholes
        # and logs create different visual/handling hazards.
        counts = {1: 12, 2: 22, 3: 34}
        type_weights = {
            1: [('ROCK', 0.60), ('POTHOLE', 0.40)],
            2: [('ROCK', 0.45), ('POTHOLE', 0.30), ('LOG', 0.15), ('BOULDER', 0.10)],
            3: [('ROCK', 0.35), ('POTHOLE', 0.30), ('LOG', 0.20), ('BOULDER', 0.15)]
        }
        self.obstacles=[]
        attempts=0
        while len(self.obstacles)<counts[self.difficulty] and attempts<1500:
            attempts+=1
            x=rng.uniform(-20,20); y=rng.uniform(-20,20)
            kind=rng.choices([k for k,_ in type_weights[self.difficulty]],
                             weights=[w for _,w in type_weights[self.difficulty]], k=1)[0]
            if kind == 'POTHOLE':
                w=h=rng.uniform(2.0,3.6)
            elif kind == 'BOULDER':
                w=h=rng.uniform(2.8,4.8)
            elif kind == 'LOG':
                w=rng.uniform(3.5,5.5); h=rng.uniform(1.2,1.8)
            else:
                w=rng.uniform(2.2,4.0); h=rng.uniform(2.2,4.0)
            clearance=6 if kind in ('POTHOLE','ROCK','BOULDER') else 5
            if math.hypot(x+22,y+22)<clearance or math.hypot(x-22,y-22)<clearance: continue
            if any(abs(x-o.x)<(w+o.w)/2+1.2 and abs(y-o.y)<(h+o.h)/2+1.2 for o in self.obstacles): continue
            self.obstacles.append(Obstacle(x,y,w,h,kind))

    @property
    def x(self): return self.ugv['x']
    @property
    def y(self): return self.ugv['y']

    def circle_rect_collision(self,x,y,r=0.8):
        # Solid obstacles: rocks, boulders and logs. Potholes are terrain
        # hazards handled separately so the UGV can drive through them.
        for o in self.obstacles:
            if o.kind == 'POTHOLE':
                continue
            nx=max(o.x-o.w/2,min(x,o.x+o.w/2)); ny=max(o.y-o.h/2,min(y,o.y+o.h/2))
            if (x-nx)**2+(y-ny)**2<r*r: return True
        return False

    def pothole_hit(self,x,y):
        for o in self.obstacles:
            if o.kind != 'POTHOLE':
                continue
            rx=max(0.7,o.w*0.40); ry=max(0.7,o.h*0.40)
            if ((x-o.x)/rx)**2 + ((y-o.y)/ry)**2 <= 1.0:
                return o
        return None

    def register_hazard_hit(self, kind):
        self.hazard_hits += 1
        self.hazard_hit_types[kind] = self.hazard_hit_types.get(kind, 0) + 1
        self.status = f'{kind} HIT — TERRAIN RESPONSE'

    def pothole_segment_hit(self, x1, y1, x2, y2):
        """Return the pothole touched while moving from (x1,y1) to (x2,y2).

        The previous version called this method but did not define it, which
        caused the AttributeError shown in PowerShell.  We test the complete
        movement segment rather than only the final UGV position, so a fast
        movement step cannot jump over a pothole.
        """
        dx = x2 - x1
        dy = y2 - y1
        for o in self.obstacles:
            if o.kind != 'POTHOLE':
                continue

            # Expand the pothole slightly by the UGV collision radius.
            # This makes contact with the edge count as entering the hazard.
            rx = max(0.75, o.w * 0.40) + 0.75
            ry = max(0.75, o.h * 0.40) + 0.75

            # Transform the ellipse into a unit circle and solve whether the
            # movement segment intersects that circle.
            ax = (x1 - o.x) / rx
            ay = (y1 - o.y) / ry
            bx = dx / rx
            by = dy / ry

            a = bx * bx + by * by
            c = ax * ax + ay * ay - 1.0

            if c <= 0.0:
                return o

            if a <= 1e-12:
                continue

            b = 2.0 * (ax * bx + ay * by)
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                continue

            root = math.sqrt(disc)
            t1 = (-b - root) / (2.0 * a)
            t2 = (-b + root) / (2.0 * a)
            if 0.0 <= t1 <= 1.0 or 0.0 <= t2 <= 1.0:
                return o

        return None

    def raycast(self, angle_deg, max_range=18):
        a=math.radians(self.ugv['heading']+angle_deg)
        step=0.22
        d=0
        while d<=max_range:
            px=self.x+math.cos(a)*d; py=self.y+math.sin(a)*d
            if abs(px)>MAP/2 or abs(py)>MAP/2 or self.circle_rect_collision(px,py,0.35): return d
            d+=step
        return max_range

    def update_lidar(self):
        self.lidar=[(a,self.raycast(a)) for a in [-75,-50,-25,0,25,50,75]]

    def autopilot(self, dt):
        if self.escape_timer>0:
            self.escape_timer-=dt
            self.ugv['speed']=-5.0
            self.ugv['heading'] += self.escape_dir*95*dt
            return
        front=dict(self.lidar).get(0,18); fl=dict(self.lidar).get(-25,18); fr=dict(self.lidar).get(25,18)
        left=dict(self.lidar).get(-50,18); right=dict(self.lidar).get(50,18)
        gx,gy=self.goal
        target=math.degrees(math.atan2(gy-self.y,gx-self.x))
        err=(target-self.ugv['heading']+180)%360-180
        avoid=(max(0,4.8-fl)+max(0,3.8-left))-(max(0,4.8-fr)+max(0,3.8-right))
        steer=max(-70,min(70,err*0.85+avoid*18))
        if front<2.5:
            self.escape_dir = 1 if right>left else -1
            self.escape_timer=0.75
            self.status='AUTOPILOT: ESCAPE / REPLAN'
            return
        self.ugv['heading'] += steer*dt
        speed=5.8
        if front<5: speed=3.2
        if abs(err)>45: speed=2.0
        self.ugv['speed']=speed
        self.status='AUTOPILOT: NAVIGATING'

    def manual(self, keys, dt):
        throttle=(keys[pygame.K_w] or keys[pygame.K_UP])-(keys[pygame.K_s] or keys[pygame.K_DOWN])
        steer=(keys[pygame.K_d] or keys[pygame.K_RIGHT])-(keys[pygame.K_a] or keys[pygame.K_LEFT])
        target=throttle*7.0
        self.ugv['speed'] += (target-self.ugv['speed'])*min(1,dt*5)
        if abs(self.ugv['speed'])>0.2: self.ugv['heading'] += steer*95*dt*(0.35+abs(self.ugv['speed'])/7)
        if not throttle: self.ugv['speed']*=0.94
        self.status='MANUAL CONTROL'

    def move(self,dt, keys=None):
        # Potholes behave like a temporary stuck state: forward motion is
        # blocked, but the operator can reverse out of the pothole.  This is
        # closer to a real UGV recovery action than forcing a full reset.
        if getattr(self, 'stuck', False):
            reverse = bool(keys and (keys[pygame.K_s] or keys[pygame.K_DOWN]))
            if not reverse or self.mode != 'MANUAL':
                self.ugv['speed'] = 0
                self.status = 'TERRAIN HAZARD — STUCK | REVERSE TO ESCAPE'
                return

            h = math.radians(self.ugv['heading'])
            # Controlled reverse speed for recovery.
            reverse_speed = -3.5
            nx = self.ugv['x'] + math.cos(h) * reverse_speed * dt
            ny = self.ugv['y'] + math.sin(h) * reverse_speed * dt

            # Solid obstacles still block the recovery movement.
            if self.circle_rect_collision(nx, ny):
                self.ugv['speed'] = 0
                self.status = 'STUCK — REVERSE BLOCKED'
                return

            self.ugv['x'] = max(-MAP/2+1, min(MAP/2-1, nx))
            self.ugv['y'] = max(-MAP/2+1, min(MAP/2-1, ny))
            self.ugv['speed'] = reverse_speed

            # Once the UGV is completely outside the pothole, recovery is
            # complete and normal driving is restored.
            if self.pothole_hit(self.ugv['x'], self.ugv['y']) is None:
                self.stuck = False
                self._in_pothole = False
                self.ugv['speed'] = 0
                self.status = 'RECOVERED — CONTINUE DRIVING'
            else:
                self.status = 'TERRAIN HAZARD — REVERSING OUT'
            return
        h=math.radians(self.ugv['heading'])
        ox, oy = self.ugv['x'], self.ugv['y']
        nx=ox+math.cos(h)*self.ugv['speed']*dt; ny=oy+math.sin(h)*self.ugv['speed']*dt
        # Check the whole movement segment, not only the final position. This
        # prevents a fast UGV from visually sliding across a pothole between frames.
        pothole_on_path = self.pothole_segment_hit(ox, oy, nx, ny)
        if pothole_on_path:
            self.register_hazard_hit('POTHOLE')
            self.stuck = True
            self.ugv['speed'] = 0
            self.status = 'TERRAIN HAZARD — UGV STUCK'
            self._in_pothole = True
            return
        if self.circle_rect_collision(nx,ny):
            self.collisions+=1
            self.ugv['speed']=0
            if self.mode=='AUTOPILOT':
                self.escape_dir = 1 if self.raycast(60)>self.raycast(-60) else -1
                self.escape_timer=0.9
                self.status='AUTOPILOT: OBSTACLE COLLISION — REPLANNING'
            else: self.status='OBSTACLE COLLISION DETECTED'
        else:
            self.ugv['x']=max(-MAP/2+1,min(MAP/2-1,nx)); self.ugv['y']=max(-MAP/2+1,min(MAP/2-1,ny))
            pothole=self.pothole_hit(self.ugv['x'],self.ugv['y'])
            if pothole:
                # Only count a new hit when the UGV first enters the pothole.
                if not getattr(self, '_in_pothole', False):
                    self.register_hazard_hit('POTHOLE')
                    self.stuck = True
                    self.ugv['speed'] = 0
                    self.status = 'TERRAIN HAZARD — UGV STUCK'
                self._in_pothole=True
            else:
                self._in_pothole=False
        if math.hypot(self.ugv['x']-self.goal[0],self.ugv['y']-self.goal[1])<2.0 and not self.success:
            self.success=True; self.ugv['speed']=0; self.log_result()

    def finish_run(self, success, reason):
        if self.current_run_logged:
            return
        self.current_run_logged = True
        elapsed = time.time() - self.run_start
        goal_dist = math.hypot(self.ugv['x']-self.goal[0], self.ugv['y']-self.goal[1])
        self.run_score = self.ai.score_run(success, elapsed, self.collisions, self.difficulty)
        new = not self.log_path.exists()
        with self.log_path.open('a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['timestamp','seed','difficulty','mode','adaptive_mode',
                            'time_sec','collisions','hazard_hits','rock_hits','pothole_hits','log_hits','boulder_hits','success','goal_distance','score','reason'])
            w.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), self.seed, self.difficulty,
                        self.mode, int(self.adaptive_mode), round(elapsed,2), self.collisions, self.hazard_hits,
                        self.hazard_hit_types.get('ROCK',0), self.hazard_hit_types.get('POTHOLE',0),
                        self.hazard_hit_types.get('LOG',0), self.hazard_hit_types.get('BOULDER',0),
                        int(success), round(goal_dist,2), round(self.run_score,1), reason])
        self.refresh_results()
        self.status = reason
        if self.adaptive_mode:
            self.difficulty = self.ai.recommended

    def log_result(self):
        self.finish_run(True, 'MISSION SUCCESS — ADAPTIVE RESULT LOGGED')

    def new_terrain(self):
        self.seed=random.randint(1,999999)
        self.reset_run()

    def world_poly(self, pts, z=0): return [self.camera.project(x,y,z)[:2] for x,y in pts]

    def draw_world(self):
        self.screen.fill((24,32,38))
        # layered ground plane
        corners=self.world_poly([(-30,-30),(30,-30),(30,30),(-30,30)])
        pygame.draw.polygon(self.screen,(55,78,60),corners)
        # grid
        for v in range(-30,31,5):
            a=self.world_poly([(v,-30),(v,30)]); b=self.world_poly([(-30,v),(30,v)])
            pygame.draw.line(self.screen,(83,108,83),a[0],a[1],1)
            pygame.draw.line(self.screen,(83,108,83),b[0],b[1],1)
        # Obstacles with distinct visual types: rocks, potholes, logs and boulders.
        obs=[]
        for o in self.obstacles:
            depth=self.camera.project(o.x,o.y)[3]
            obs.append((depth,o))
        for _,o in sorted(obs,reverse=True):
            cx,cy,sc=self.camera.project(o.x,o.y,0)[:3]
            if o.kind == 'POTHOLE':
                # Dark depression with an irregular rim.
                pts=[]
                for i in range(12):
                    a=2*math.pi*i/12
                    rr=0.82+0.10*math.sin(i*2.7+self.seed)
                    pts.append(self.camera.project(o.x+math.cos(a)*o.w*0.46*rr,
                                                   o.y+math.sin(a)*o.h*0.46*rr,0.05)[:2])
                pygame.draw.polygon(self.screen,(38,31,25),pts)
                rim=[]
                for i in range(12):
                    a=2*math.pi*i/12
                    rim.append(self.camera.project(o.x+math.cos(a)*o.w*0.52,
                                                   o.y+math.sin(a)*o.h*0.52,0.08)[:2])
                pygame.draw.polygon(self.screen,(105,84,57),rim,3)
            elif o.kind == 'LOG':
                # Low horizontal log with circular ends.
                pts=[(o.x-o.w/2,o.y-o.h/2),(o.x+o.w/2,o.y-o.h/2),
                     (o.x+o.w/2,o.y+o.h/2),(o.x-o.w/2,o.y+o.h/2)]
                top=self.world_poly(pts,1.0); base=self.world_poly(pts,0)
                pygame.draw.polygon(self.screen,(105,63,35),base)
                pygame.draw.polygon(self.screen,(151,91,47),top)
                pygame.draw.line(self.screen,(70,42,25),top[0],top[1],3)
            else:
                # Rock/boulder: stacked irregular polygons for a 3-D effect.
                radius=min(o.w,o.h)*0.48
                pts=[]
                top=[]
                for i in range(8):
                    a=2*math.pi*i/8
                    rr=radius*(0.82+0.16*math.sin(i*1.9+self.seed))
                    px=o.x+math.cos(a)*rr; py=o.y+math.sin(a)*rr*0.88
                    pts.append((px,py)); top.append((px,py))
                base=self.world_poly(pts,0.0); upper=self.world_poly([(x,y) for x,y in top],1.8 if o.kind=='BOULDER' else 1.25)
                pygame.draw.polygon(self.screen,(76,67,59),base)
                pygame.draw.polygon(self.screen,(137,126,111),upper)
                for i in range(len(base)):
                    pygame.draw.line(self.screen,(92,82,72),base[i],upper[i],2)

        # goal zone and start zone
        for (gx,gy,label,fill) in [(*self.goal,'GOAL',(45,180,70)),(-22,-22,'START',(35,160,190))]:
            pts=[(gx-2.5,gy-2.5),(gx+2.5,gy-2.5),(gx+2.5,gy+2.5),(gx-2.5,gy+2.5)]
            pygame.draw.polygon(self.screen,fill,self.world_poly(pts))
            p=self.world_poly(pts,2.5)
            for q in p: pygame.draw.line(self.screen,(240,240,90),q,(q[0],q[1]-18),3)
            self.text(label,(self.camera.project(gx,gy,2.7)[0],self.camera.project(gx,gy,2.7)[1]-35),self.small)

        self.draw_lidar()
        self.draw_ugv()

    def draw_lidar(self):
        for ang,d in self.lidar:
            a=math.radians(self.ugv['heading']+ang)
            end=(self.x+math.cos(a)*d,self.y+math.sin(a)*d)
            p1=self.camera.project(self.x,self.y,1.0)[:2]; p2=self.camera.project(*end,0.8)[:2]
            pygame.draw.line(self.screen,(245,215,60),p1,p2,2)
            pygame.draw.circle(self.screen,(255,230,90),(int(p2[0]),int(p2[1])),3)

    def draw_ugv(self):
        h=math.radians(self.ugv['heading']); c=math.cos(h); s=math.sin(h)
        def tr(px,py): return (self.x+c*px-s*py,self.y+s*px+c*py)
        body=[tr(1.5,0.95),tr(1.5,-0.95),tr(-1.5,-0.95),tr(-1.5,0.95)]
        bp=self.world_poly(body,0.9)
        pygame.draw.polygon(self.screen,(42,103,190),bp); pygame.draw.polygon(self.screen,(100,170,235),self.world_poly(body,2.0))
        for wx in [-1.05,1.05]:
            for wy in [-1.05,1.05]:
                p=tr(wx,wy); q=self.camera.project(*p,0.55)[:2]
                pygame.draw.circle(self.screen,(25,25,28),(int(q[0]),int(q[1])),max(5,int(self.camera.project(*p,0.55)[2]*0.32)))
        nose=tr(1.9,0); q=self.camera.project(*nose,2.0)[:2]; pygame.draw.circle(self.screen,(250,220,60),(int(q[0]),int(q[1])),7)

    def text(self,s,pos,font=None,anchor='topleft'):
        surf=(font or self.font).render(str(s),True,(235,245,250)); r=surf.get_rect()
        x, y = int(pos[0]), int(pos[1])
        # pygame.Rect anchor attributes such as `left` and `center` have
        # different value types. Map our friendly anchor names explicitly.
        if anchor == 'left':
            anchor = 'topleft'
        if anchor == 'center':
            r.center = (x, y)
        elif anchor == 'midright':
            r.midright = (x, y)
        elif anchor == 'midbottom':
            r.midbottom = (x, y)
        elif anchor == 'midtop':
            r.midtop = (x, y)
        elif anchor == 'topright':
            r.topright = (x, y)
        elif anchor == 'bottomleft':
            r.bottomleft = (x, y)
        elif anchor == 'bottomright':
            r.bottomright = (x, y)
        else:
            r.topleft = (x, y)
        self.screen.blit(surf,r)

    def draw_hud(self):
        pygame.draw.rect(self.screen,(8,13,17),(0,0,W,112))
        self.text('UGV ADAPTIVE TRAINING SIMULATOR',(22,18),self.title)
        self.text(f"Difficulty: {self.difficulty}   Mode: {self.mode}   Collisions: {self.collisions}   Terrain Hits: {self.hazard_hits}   AI: {'ON' if self.adaptive_mode else 'OFF'}",(22,63),self.font)
        self.text(f'Status: {self.status}',(22,88),self.small)
        ai_panel=pygame.Rect(W-390,166,375,112)
        pygame.draw.rect(self.screen,(10,18,23),ai_panel); pygame.draw.rect(self.screen,(120,150,165),ai_panel,2)
        self.text('AI ADAPTIVE TRAINING',(W-375,177),self.font)
        self.text(f'Skill score: {self.ai.skill_score():.0f}/100',(W-375,211),self.small)
        self.text(f'Runs analyzed: {len(self.ai.history)}',(W-375,234),self.small)
        self.text(f'Recommendation: D{self.ai.recommended}',(W-375,255),self.small)
        panel=pygame.Rect(W-335,14,320,140)
        pygame.draw.rect(self.screen,(10,18,23),panel); pygame.draw.rect(self.screen,(110,140,150),panel,2)
        self.text('MOUSE 3D CAMERA',(W-320,25),self.font)
        self.text('LMB drag: orbit   RMB drag: pan',(W-320,59),self.small)
        self.text('Wheel: zoom   F: reset camera',(W-320,82),self.small)
        self.text('C: follow/top   M: autopilot',(W-320,105),self.small)
        # bottom controls
        pygame.draw.rect(self.screen,(8,13,17),(0,H-58,W,58))
        self.text('WASD drive | 1/2/3 difficulty | N terrain | T adaptive | M autopilot | R reset after hazard/collision | ESC quit', (20,H-43),self.small)
        # lidar values
        vals='LiDAR: '+'  '.join(f'{a:+d}° {d:4.1f}m' for a,d in self.lidar)
        self.text(vals,(20,H-75),pygame.font.SysFont('consolas',15,bold=True))
        if self.success:
            self.text(f'MISSION SUCCESS — SCORE {self.run_score:.0f}',(W/2, H/2),self.big,'center')
        elif self.current_run_logged and self.run_score is not None:
            self.text(f'TIME LIMIT — SCORE {self.run_score:.0f}',(W/2, H/2),self.big,'center')


    def panel(self, rect, title=None, accent=(75, 170, 220), fill=(12, 20, 26), radius=10):
        pygame.draw.rect(self.screen, fill, rect, border_radius=radius)
        pygame.draw.rect(self.screen, (55, 75, 85), rect, 1, border_radius=radius)
        if title:
            self.text(title, (rect.x+16, rect.y+12), self.small)
        pygame.draw.rect(self.screen, accent, (rect.x, rect.y, 4, rect.h), border_radius=2)

    def draw_line_chart(self, rect, values, title, max_value=100, suffix=''):
        self.panel(rect, title)
        if not values:
            self.text('No completed runs yet', rect.center, self.small, 'center')
            return
        chart=pygame.Rect(rect.x+48, rect.y+48, rect.w-66, rect.h-76)
        # grid
        for k in range(5):
            y=chart.bottom-k*chart.h/4
            pygame.draw.line(self.screen,(38,55,63),(chart.left,y),(chart.right,y),1)
            label=f'{max_value*k/4:.0f}{suffix}'
            self.text(label,(chart.left-8,y),pygame.font.SysFont('consolas',11),'midright')
        vals=[max(0,min(float(max_value),float(v))) for v in values]
        pts=[]
        for i,v in enumerate(vals):
            x=chart.left if len(vals)==1 else chart.left+i*(chart.w/(len(vals)-1))
            y=chart.bottom-(v/max_value)*chart.h
            pts.append((x,y))
        if len(pts)>1:
            pygame.draw.lines(self.screen,(65,180,235),False,pts,3)
        for i,p in enumerate(pts):
            pygame.draw.circle(self.screen,(235,245,250),(int(p[0]),int(p[1])),5)
            if len(vals)<=10:
                self.text(f'{vals[i]:.0f}{suffix}',(p[0],p[1]-10),pygame.font.SysFont('consolas',11),'midbottom')

    def draw_bar_chart(self, rect, values, title, max_value=None, suffix=''):
        self.panel(rect, title)
        if not values:
            self.text('No completed runs yet', rect.center, self.small, 'center')
            return
        vals=[float(v) for v in values]
        mv=max_value if max_value is not None else max(1.0,max(vals))
        chart=pygame.Rect(rect.x+42,rect.y+48,rect.w-58,rect.h-72)
        for k in range(1,4):
            y=chart.bottom-chart.h*k/4
            pygame.draw.line(self.screen,(38,55,63),(chart.left,y),(chart.right,y),1)
        n=len(vals); gap=6; bw=max(8,(chart.w-gap*(n-1))//n)
        for i,v in enumerate(vals):
            bh=int(max(0,min(1,v/mv))*chart.h)
            x=chart.left+i*(bw+gap); y=chart.bottom-bh
            pygame.draw.rect(self.screen,(65,155,210),(x,y,bw,bh),border_radius=3)
            if n<=12:
                self.text(f'{v:.0f}{suffix}',(x+bw/2,y-3),pygame.font.SysFont('consolas',10),'midbottom')

    def draw_donut(self, center, radius, success_count, total):
        if total<=0:
            pygame.draw.circle(self.screen,(35,50,58),center,radius,18)
            return
        rect=pygame.Rect(center[0]-radius,center[1]-radius,radius*2,radius*2)
        ratio=success_count/total
        pygame.draw.arc(self.screen,(44,66,75),rect,-math.pi/2,3*math.pi/2,18)
        pygame.draw.arc(self.screen,(65,190,110),rect,-math.pi/2,-math.pi/2+2*math.pi*ratio,18)
        self.text(f'{ratio*100:.0f}%',center,self.big,'center')
        self.text('SUCCESS', (center[0],center[1]+30),self.small,'midtop')

    def _draw_dashboard_surface(self):
        # Professional full-screen analytics workspace
        self.screen.fill((7, 12, 17))
        rows=self.all_results
        scores=[float(r.get('score',0)) for r in rows]
        collisions=[float(r.get('collisions',0)) for r in rows]
        hazard_hits=[float(r.get('hazard_hits',0)) for r in rows]
        times=[float(r.get('time_sec',0)) for r in rows]
        successes=[str(r.get('success','0'))=='1' for r in rows]
        difficulties=[int(float(r.get('difficulty',0))) for r in rows]
        n=len(rows); succ=sum(successes)
        avg_score=sum(scores)/n if n else 0
        avg_col=sum(collisions)/n if n else 0
        avg_time=sum(times)/n if n else 0
        best_score=max(scores) if scores else 0
        latest=rows[-1] if rows else None

        # top bar
        pygame.draw.rect(self.screen,(10,18,24),(0,0,W,86))
        pygame.draw.rect(self.screen,(65,165,220),(0,84,W,2))
        self.text('TRAINING ANALYTICS', (28,16), self.title)
        self.text('UGV ADAPTIVE TRAINING  /  PERFORMANCE CENTER', (30,57), self.small)
        self.text('P  Close', (W-145,24), self.font)
        self.text(f'CSV  {self.log_path.name}  |  {len(rows)} records', (W-330,58), pygame.font.SysFont('consolas',13))

        # KPI cards
        cards=[
            ('TRAINING RUNS',f'{n}', 'Recorded sessions', (65,155,220)),
            ('SUCCESS RATE',f'{succ/n*100:.0f}%' if n else '--','Goal completion',(65,190,115)),
            ('AVG SKILL',f'{avg_score:.0f}/100' if n else '--','Adaptive score',(190,130,235)),
            ('AVG COLLISIONS',f'{avg_col:.1f}' if n else '--','Rock / log impacts',(235,165,70)),
            ('HAZARD HITS',f'{sum(hazard_hits):.0f}' if n else '--','Pothole / terrain hits',(220,125,80)),
            ('BEST SCORE',f'{best_score:.0f}' if n else '--','Personal best',(70,195,205)),
        ]
        x=28
        for title,val,sub,accent in cards:
            r=pygame.Rect(x,102,190,92)
            self.panel(r,fill=(11,20,27),accent=accent)
            self.text(title,(x+16,114),pygame.font.SysFont('segoeui',13,bold=True))
            self.text(val,(x+16,137),self.big)
            self.text(sub,(x+16,174),pygame.font.SysFont('segoeui',12))
            x+=205

        # main charts
        self.draw_line_chart(pygame.Rect(28,214,565,225),scores[-12:],'SKILL SCORE TREND',100)
        self.draw_line_chart(pygame.Rect(608,214,565,225),times[-12:],'COMPLETION TIME',90,'s')

        # success / AI panel
        r=pygame.Rect(1188,214,64,225)
        pygame.draw.rect(self.screen,(11,20,27),r,border_radius=10)
        # narrow vertical indicator
        self.text('AI',(1220,230),pygame.font.SysFont('segoeui',14,bold=True),'center')
        self.text('D'+str(self.ai.recommended),(1220,330),self.big,'center')
        self.text('NEXT',(1220,370),pygame.font.SysFont('segoeui',12,bold=True),'center')

        # lower section
        # Lower cards are laid out with fixed gaps so they never overlap.
        self.panel(pygame.Rect(28,459,280,250),'SUCCESS OVERVIEW',(65,190,115))
        self.draw_donut((168,566),68,succ,n)
        self.text(f'{succ} successful / {n-succ} incomplete',(168,657),self.small,'center')
        self.text('Target: complete the mission with', (168,681),pygame.font.SysFont('segoeui',12),'center')
        self.text('minimal collisions and time.', (168,698),pygame.font.SysFont('segoeui',12),'center')

        self.draw_bar_chart(pygame.Rect(320,459,240,250),collisions[-12:],'SOLID COLLISIONS',max(1,max(collisions,default=1)))
        self.draw_bar_chart(pygame.Rect(572,459,240,250),hazard_hits[-12:],'TERRAIN HITS',max(1,max(hazard_hits,default=1)))
        self.draw_bar_chart(pygame.Rect(824,459,200,250),difficulties[-12:],'DIFFICULTY',3)

        # AI coach card -- separate column with enough width for all text.
        r=pygame.Rect(1036,459,240,250)
        self.panel(r,'AI COACH',(190,130,235))
        self.text('ADAPTIVE STATUS',(r.x+16,r.y+48),pygame.font.SysFont('segoeui',11,bold=True))
        self.text('ACTIVE' if self.adaptive_mode else 'PAUSED',(r.x+16,r.y+67),self.font)
        self.text('Skill',(r.x+16,r.y+108),self.small)
        self.text(f'{self.ai.skill_score():.0f}/100',(r.x+16,r.y+128),self.font)
        self.text('Recommendation',(r.x+16,r.y+164),self.small)
        self.text(f'D{self.ai.recommended}',(r.x+16,r.y+184),self.big)
        # bottom strip / recent runs
        table=pygame.Rect(28,727,1248,235)
        self.panel(table,'RECENT TRAINING RUNS',(70,195,205))
        headers=['RUN','DIFFICULTY','MODE','SCORE','COLLISIONS','HAZARDS','TIME','RESULT']
        colx=[40,135,260,370,480,600,710,820]
        for x,h in zip(colx[1:],headers):
            self.text(h,(x,766),pygame.font.SysFont('consolas',11,bold=True))
        recent=rows[-8:]
        if recent:
            for j,rw in enumerate(reversed(recent)):
                y=790+j*28
                res='SUCCESS' if str(rw.get('success','0'))=='1' else 'INCOMPLETE'
                vals=[f"#{n-j}",f"D{rw.get('difficulty','-')}",str(rw.get('mode','-')),str(rw.get('score','-')),str(rw.get('collisions','-')),str(rw.get('hazard_hits','0')),f"{float(rw.get('time_sec',0)):.1f}s",res]
                for x,v in zip(colx,vals):
                    self.text(v,(x,y),pygame.font.SysFont('segoeui',13,bold=(x==45)), 'left')
        else:
            self.text('No training records yet. Complete a mission to populate this table.',(45,805),self.small)

    def draw_dashboard(self):
        # Render the analytics dashboard on a taller surface so the user can
        # scroll through the full Recent Runs section with the mouse wheel.
        visible_h = self.screen.get_height()
        content_h = max(980, visible_h + 260)
        old_screen = self.screen
        self.screen = pygame.Surface((old_screen.get_width(), content_h))
        self._draw_dashboard_surface()
        # Keep the top bar fixed while scrolling the dashboard content.
        old_screen.fill((7, 12, 17))
        old_screen.blit(self.screen, (0, 0), pygame.Rect(0, 0, old_screen.get_width(), 86))
        max_scroll = max(0, content_h - visible_h)
        scroll = max(0, min(self.dashboard_scroll, max_scroll))
        if visible_h > 86:
            old_screen.blit(self.screen, (0, 86), pygame.Rect(0, 86 + scroll, old_screen.get_width(), visible_h - 86))
        self.screen = old_screen

    def run(self):
        running=True
        while running:
            dt=self.clock.tick(FPS)/1000
            for e in pygame.event.get():
                if e.type==pygame.QUIT: running=False
                elif e.type==pygame.MOUSEBUTTONDOWN:
                    if self.dashboard and e.button in (4,5):
                        self.dashboard_scroll += -60 if e.button == 4 else 60
                        self.dashboard_scroll = max(0, self.dashboard_scroll)
                    elif e.button in (1,3):
                        self.camera.begin_drag(e.button,e.pos)
                    elif e.button==4: self.camera.zoom(1)
                    elif e.button==5: self.camera.zoom(-1)
                elif e.type==pygame.MOUSEWHEEL:
                    if self.dashboard:
                        self.dashboard_scroll = max(0, self.dashboard_scroll + (-e.y*60))
                    else:
                        self.camera.zoom(e.y)
                elif e.type==pygame.MOUSEBUTTONUP and e.button in (1,3): self.camera.end_drag(e.button)
                elif e.type==pygame.MOUSEMOTION: self.camera.mouse_motion(e.pos)
                elif e.type==pygame.KEYDOWN:
                    if e.key==pygame.K_ESCAPE: running=False
                    elif e.key==pygame.K_r: self.reset_run()
                    elif e.key==pygame.K_n: self.new_terrain()
                    elif e.key in (pygame.K_1,pygame.K_2,pygame.K_3): self.difficulty=int(e.unicode); self.reset_run()
                    elif e.key==pygame.K_m: self.mode='AUTOPILOT' if self.mode=='MANUAL' else 'MANUAL'; self.status=self.mode
                    elif e.key==pygame.K_t:
                        self.adaptive_mode = not self.adaptive_mode
                        self.status = 'ADAPTIVE AI: ON' if self.adaptive_mode else 'ADAPTIVE AI: OFF'
                    elif e.key==pygame.K_f: self.camera.reset()
                    elif e.key==pygame.K_c: self.camera.follow=not self.camera.follow
                    elif e.key==pygame.K_p: self.dashboard=not self.dashboard; self.dashboard_scroll=0; self.refresh_results()
            if not self.dashboard:
                if not self.success:
                    self.elapsed=time.time()-self.run_start
                    if self.elapsed >= self.max_run_time:
                        self.ugv['speed'] = 0
                        self.finish_run(False, 'TIME LIMIT — ADAPTIVE COACHING APPLIED')
                    self.update_lidar()
                    if not self.current_run_logged:
                        if self.mode=='AUTOPILOT': self.autopilot(dt)
                        else: self.manual(pygame.key.get_pressed(),dt)
                        self.move(dt, pygame.key.get_pressed())
                self.camera.update(self.ugv)
                self.draw_world(); self.draw_hud()
            else:
                self.draw_dashboard()
            pygame.display.flip()
        if not self.dashboard and not self.current_run_logged:
            self.finish_run(False, 'SESSION ENDED — ATTEMPT LOGGED')
        pygame.quit()

if __name__=='__main__': Simulator().run()
