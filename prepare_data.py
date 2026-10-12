import sqlite3

def update_traffic_labels(db_path="data.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Проверяем наличие столбца traffic_level, при необходимости добавляем
    cursor.execute("PRAGMA table_info(traffic)")
    columns = [col[1] for col in cursor.fetchall()]
    if "traffic_level" not in columns:
        cursor.execute("ALTER TABLE traffic ADD COLUMN traffic_level INTEGER")

    cursor.execute("SELECT id, temperature, cars, light FROM traffic")
    rows = cursor.fetchall()

    for row_id, temperature, cars, light in rows:
        weather_factor = 1.5 if (temperature < -10 or temperature > 35) else 1.0
        light_factor = 1.2 if light == 1 else 0.7
        effective_cars = cars * weather_factor * light_factor

        if effective_cars < 20:
            traffic_level = 0
        elif effective_cars < 50:
            traffic_level = 1
        else:
            traffic_level = 2

        cursor.execute(
            "UPDATE traffic SET traffic_level = ? WHERE id = ?",
            (traffic_level, row_id)
        )

    conn.commit()
    conn.close()

if __name__ == "__main__":
    update_traffic_labels()