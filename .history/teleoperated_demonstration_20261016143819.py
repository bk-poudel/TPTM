def main():
    global running
    print("A: Starting script", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        OUTPUT_DIR, f"fr3_demo_{timestamp}.csv"
    )
    sim = None
    logger = None
    listener = None
    try:
        print("B: Creating FR3Sim", flush=True)
        sim = FR3Sim(
            interface_type="torque",
            render=True,
            dt=DT,
            gravity=True,
        )
        print("C: FR3Sim created", flush=True)
        # Read the initial pose BEFORE starting the keyboard listener.
        print("D: Reading joint state", flush=True)
        q, dq = sim.get_state()
        q = np.asarray(q).copy()
        dq = np.asarray(dq).copy()
        print("E: Joint state read", flush=True)
        print("F: Reading end-effector pose", flush=True)
        T = sim.get_pose(q)
        print("G: End-effector pose read", flush=True)
        target_position = T[:3, 3].copy()
        target_rotation = T[:3, :3].copy()
        print("Initial EE position:", target_position, flush=True)
        # Start the logger only after pose initialization succeeds.
        logger = CSVLogger(csv_path)
        logger.start()
        print("H: Logger started", flush=True)
        print("I: Starting keyboard listener", flush=True)
        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
        )
        listener.start()
        print("J: Keyboard listener started", flush=True)
        previous_velocity = np.zeros(6)
        sim_time = 0.0
        recording_start = None
        print("\nTeleoperation started.", flush=True)
        print("W/S: X | A/D: Y | R/F: Z", flush=True)
        print("U/O: Roll | I/K: Pitch | J/L: Yaw", flush=True)
        print("SPACE: Pause | ENTER: Record | ESC: Quit", flush=True)
        print("CSV:", csv_path, flush=True)
        while running and sim.viewer.is_running():
            # Snapshot keyboard state.
            with key_lock:
                keys = set(pressed_keys)
            if motion_enabled:
                direction = np.array([
                    float("w" in keys) - float("s" in keys),
                    float("d" in keys) - float("a" in keys),
                    float("r" in keys) - float("f" in keys),
                ])
                target_position += (
                    TRANSLATION_SPEED * DT * direction
                )
                target_position[0] = np.clip(
                    target_position[0], *X_LIMITS
                )
                target_position[1] = np.clip(
                    target_position[1], *Y_LIMITS
                )
                target_position[2] = np.clip(
                    target_position[2], *Z_LIMITS
                )
                angular_direction = np.array([
                    float("u" in keys) - float("o" in keys),
                    float("i" in keys) - float("k" in keys),
                    float("j" in keys) - float("l" in keys),
                ])
                delta_rotation = Rotation.from_rotvec(
                    ROTATION_SPEED * DT * angular_direction
                ).as_matrix()
                target_rotation = (
                    delta_rotation @ target_rotation
                )
            q, dq = sim.get_state()
            q = np.asarray(q).copy()
            dq = np.asarray(dq).copy()
            tau, T, velocity = cartesian_impedance_torque(
                sim, q, dq, target_position, target_rotation
            )
            sim.send_joint_torque(tau, finger_pos=0.035)
            sim_time += DT
            q, dq = sim.get_state()
            q = np.asarray(q).copy()
            dq = np.asarray(dq).copy()
            T = sim.get_pose(q)
            J = sim.get_jacobian(q)
            position = T[:3, 3].copy()
            rotation = T[:3, :3].copy()
            rpy = Rotation.from_matrix(rotation).as_euler("xyz")
            velocity = J @ dq
            acceleration = (
                velocity - previous_velocity
            ) / DT
            previous_velocity = velocity.copy()
            if recording:
                if recording_start is None:
                    recording_start = sim_time
                recording_time = sim_time - recording_start
                row = [
                    sim_time,
                    time.time(),
                    recording_time,
                    *position,
                    *rpy,
                    *velocity[:3],
                    *velocity[3:],
                    *acceleration[:3],
                    *acceleration[3:],
                    *target_position,
                    *Rotation.from_matrix(
                        target_rotation
                    ).as_euler("xyz"),
                    *q,
                    *dq,
                    1,
                ]
                logger.log(row)
            else:
                recording_start = None
            if int(sim_time * 10) != int((sim_time - DT) * 10):
                print(
                    f"\rt={sim_time:8.2f}s | "
                    f"EE={position.round(3)} | "
                    f"recording={recording}",
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.", flush=True)
    finally:
        running = False
        if listener is not None:
            listener.stop()
        if logger is not None:
            logger.close()
            print("Dropped CSV rows:", logger.dropped_rows, flush=True)
        if sim is not None:
            sim.close()
        print("\nTeleoperation stopped.", flush=True)
        print("CSV path:", csv_path, flush=True)
if __name__ == "__main__":
    main()