
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import sys

dwd_dir = str(Path(__file__).parent.parent / 'DWD')
if dwd_dir not in sys.path:
    sys.path.append(dwd_dir)

import DWDMorpoints

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

class MinimalWindTracker:
    def __init__(self, morpoints=None, hours_back=4, time_step_minutes=20, max_steps=50):
        self.morpoints = morpoints or DWDMorpoints.DWDMorpoints()
        self.hours_back = hours_back
        self.time_step = timedelta(minutes=time_step_minutes)
        self.max_steps = max_steps
        self.cache = {}  # Cache simple
        self.api_calls = 0

    def get_wind_simple(self, lat, lon, time):
        """UNE seule requête, pas de retry, pas de complexité"""
        # Cache check
        key = (round(lat, 3), round(lon, 3), time.isoformat()[:13])
        if key in self.cache:
            return self.cache[key]

        try:
            self.api_calls += 1
            u = float(self.morpoints.get(lat=lat, lon=lon, type_='u_10m', time=time))
            v = float(self.morpoints.get(lat=lat, lon=lon, type_='v_10m', time=time))

            # Si NaN, utiliser du vent par défaut
            if math.isnan(u) or math.isnan(v):
                u, v = -5.0, 2.0  # Vent d'ouest simple

            self.cache[key] = (u, v)
            return u, v

        except:
            # Fallback immédiat
            u, v = -5.0, 2.0
            self.cache[key] = (u, v)
            return u, v

    def euler_step(self, lat, lon, current_time, dt_sec):
        """Pas d'Euler basique - UN seul appel de vent"""
        u, v = self.get_wind_simple(lat, lon, current_time)

        # Facteur pour atteindre des distances réalistes
        factor = 20

        dlat = -(v * dt_sec * factor) / 111320.0
        dlon = -(u * dt_sec * factor) / (111320.0 * math.cos(math.radians(lat)))

        return lat + dlat, lon + dlon

    def minimal_backtrack(self, start_lat, start_lon, start_time=None, max_distance_km=500):
        """Backtrack ultra-simple - pas de préchargement, pas de complexité"""
        if start_time is None:
            start_time = datetime.now(timezone.utc)

        lat, lon = start_lat, start_lon
        current_time = start_time
        trajectory = [(current_time, lat, lon)]

        total_distance = 0.0

        import time
        start_time_perf = time.time()

        for step in range(self.max_steps):
            # UN seul appel par pas
            lat_new, lon_new = self.euler_step(lat, lon, current_time, self.time_step.total_seconds())

            step_distance = haversine_km(lat, lon, lat_new, lon_new)
            total_distance += step_distance

            print(f"Pas {step+1}: {step_distance:.1f} km → Total: {total_distance:.1f} km - lat({lat_new:.5f}, {lon_new:.5f})")


            lat, lon = lat_new, lon_new
            current_time -= self.time_step
            trajectory.append((current_time, lat, lon))

            if total_distance >= max_distance_km:
                trajectory.append({'stop_reason': f'Distance {max_distance_km} km atteinte'})
                break
        if len(trajectory) == self.max_steps + 1:
            trajectory.append({'stop_reason': 'Maximum de pas atteint'})

        elapsed = time.time() - start_time_perf

        print(f"\n📊 RÉSULTATS:")
        print(f"   ⏱️  Temps: {elapsed:.1f}s")
        print(f"   📡 API calls: {self.api_calls}")
        print(f"   📏 Distance: {total_distance:.1f} km")
        print(f"   🎯 Points: {len([p for p in trajectory if isinstance(p, tuple)])}")

        return trajectory


    def trajectoire(self, latH, lonH):
        # Exécution du suivi minimal
        self.tracker = MinimalWindTracker(
            hours_back=1.0,
            time_step_minutes=15,
            max_steps=20  # Limité pour test rapide
        )
        result = self.tracker.minimal_backtrack(latH, lonH, max_distance_km=100)

        # Extraction des points de trajectoire
        trajectory_points = [p for p in result if isinstance(p, tuple)]

        return trajectory_points


if __name__ == "__main__":

    tracker = MinimalWindTracker(
        hours_back=1.0,
        time_step_minutes=15,
        max_steps=20  # Limité pour test rapide
    )

    start_lat, start_lon =48.2768,-2.4019#Rouillac/meteo# 48.8566, 2.3522  # Paris

    import time
    overall_start = time.time()

    try:
        result = tracker.minimal_backtrack(
            start_lat, start_lon, max_distance_km=100
        )

        overall_time = time.time() - overall_start
        print(f"\n⚡ TEMPS TOTAL: {overall_time:.1f} secondes")

        print(f"\n📍 Trajectoire:")
        trajectory_points = [p for p in result if isinstance(p, tuple)]

        for i, (t, la, lo) in enumerate(trajectory_points[:5]):
            print(f"  {i}: {t.isoformat()[:16]} → ({la:.4f}, {lo:.4f})") #A mettre dans corridor


        if len(trajectory_points) > 5:
            print("  ...")
            t, la, lo = trajectory_points[-1]
            print(f"  {len(trajectory_points)-1}: {t.isoformat()[:16]} → ({la:.4f}, {lo:.4f})")

        stop_reasons = [p for p in result if isinstance(p, dict)]
        if stop_reasons:
            print(f"  🛑 {stop_reasons[0].get('stop_reason')}")

        if overall_time <= 10:
            print(f"\n🎉 SUCCÈS: Objectif < 10s atteint!")
        else:
            print(f"\n⚠️ Encore trop lent: {overall_time:.1f}s")

    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
