from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime
import shutil
from yolov4 import detect_cars
from algo import optimize_traffic

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# In-memory store (replace with database in production)
activities = []
recent_activities = []  # Store recent actions
users = set()  # Track unique user IDs

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return jsonify({"message": "Server is running!"}), 200

@app.route("/upload", methods=["POST"])
def upload_files():
    """Handle video uploads and create a pending activity."""
    if "videos" not in request.files or "userId" not in request.form:
        return jsonify({"error": "Missing videos or userId"}), 400

    videos = request.files.getlist("videos")
    user_id = request.form["userId"]

    if len(videos) != 4:
        return jsonify({"error": "Exactly 4 videos are required"}), 400

    activity_id = str(uuid.uuid4())
    activity_folder = os.path.join(app.config["UPLOAD_FOLDER"], activity_id)
    os.makedirs(activity_folder)

    video_urls = []
    directions = ["north", "south", "east", "west"]
    for i, video in enumerate(videos):
        if video and allowed_file(video.filename):
            filename = secure_filename(f"{directions[i]}.mp4")
            video_path = os.path.join(activity_folder, filename)
            video.save(video_path)
            video_urls.append(f"http://localhost:5000/videos/{activity_id}/{filename}")
        else:
            shutil.rmtree(activity_folder)
            return jsonify({"error": "Invalid video file"}), 400

    # Track user
    users.add(user_id)

    # Create activity
    activity = {
        "id": activity_id,
        "userId": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        "videos": video_urls,
        "status": "pending",
        "result": None,
        "trafficCounts": None  # Initialize for later use
    }
    activities.append(activity)

    # Log recent activity
    recent_activities.append({
        "id": str(uuid.uuid4()),
        "action": f"User '{user_id}' uploaded videos",
        "timestamp": datetime.utcnow().isoformat()
    })

    return jsonify({"activityId": activity_id}), 201

@app.route("/activities", methods=["GET"])
def get_activities():
    """Return list of pending activities."""
    pending_activities = [activity for activity in activities if activity["status"] == "pending"]
    return jsonify(pending_activities), 200

@app.route("/activities/<activity_id>/approve", methods=["POST"])
def approve_activity(activity_id):
    """Approve an activity and process videos with YOLOv4 and optimization."""
    for activity in activities:
        if activity["id"] == activity_id and activity["status"] == "pending":
            video_paths = [
                os.path.join(app.config["UPLOAD_FOLDER"], activity_id, f"{direction}.mp4")
                for direction in ["north", "south", "east", "west"]
            ]
            num_cars_list = [detect_cars(video) for video in video_paths]
            result = optimize_traffic(num_cars_list)

            # Update activity with results and traffic counts
            activity["status"] = "approved"
            activity["result"] = result
            activity["trafficCounts"] = {
                "north": num_cars_list[0],
                "south": num_cars_list[1],
                "east": num_cars_list[2],
                "west": num_cars_list[3]
            }

            # Log recent activity
            recent_activities.append({
                "id": str(uuid.uuid4()),
                "action": f"Admin approved activity '{activity_id}' for user '{activity['userId']}'",
                "timestamp": datetime.utcnow().isoformat()
            })

            return jsonify({"success": True}), 200
    return jsonify({"error": "Activity not found or already processed"}), 404

@app.route("/activities/<activity_id>/reject", methods=["POST"])
def reject_activity(activity_id):
    """Reject an activity."""
    for activity in activities:
        if activity["id"] == activity_id and activity["status"] == "pending":
            activity["status"] = "rejected"

            # Log recent activity
            recent_activities.append({
                "id": str(uuid.uuid4()),
                "action": f"Admin rejected activity '{activity_id}' for user '{activity['userId']}'",
                "timestamp": datetime.utcnow().isoformat()
            })

            return jsonify({"success": True}), 200
    return jsonify({"error": "Activity not found or already processed"}), 404

@app.route("/results/<activity_id>", methods=["GET"])
def get_results(activity_id):
    """Return status and results for an activity."""
    for activity in activities:
        if activity["id"] == activity_id:
            response = {"status": activity["status"]}
            if activity["status"] == "approved" and activity["result"]:
                response["result"] = activity["result"]
            return jsonify(response), 200
    return jsonify({"error": "Activity not found"}), 404

@app.route("/videos/<activity_id>/<filename>", methods=["GET"])
def serve_video(activity_id, filename):
    """Serve video files."""
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], activity_id, filename)
    if os.path.exists(video_path):
        return send_file(video_path)
    return jsonify({"error": "Video not found"}), 404

@app.route("/dashboard/stats", methods=["GET"])
def get_dashboard_stats():
    """Return dashboard statistics."""
    return jsonify({
        "totalUsers": len(users),
        "activeSessions": 0,  # Implement session tracking if needed
        "pendingTasks": len([a for a in activities if a["status"] == "pending"]),
        "systemStatus": "Operational"  # Add health check logic if needed
    }), 200

@app.route("/dashboard/recent-activity", methods=["GET"])
def get_recent_activity():
    """Return last 10 recent activities, sorted by timestamp (newest first)."""
    sorted_activities = sorted(recent_activities, key=lambda x: x["timestamp"], reverse=True)[:10]
    return jsonify(sorted_activities), 200

@app.route("/reports", methods=["GET"])
def get_reports():
    """Return all approved activities with results and traffic counts for reporting."""
    approved_activities = [
        {
            "id": activity["id"],
            "userId": activity["userId"],
            "timestamp": activity["timestamp"],
            "trafficCounts": activity.get("trafficCounts", {
                "north": 0,
                "south": 0,
                "east": 0,
                "west": 0
            }),
            "signalTimings": activity["result"]
        }
        for activity in activities
        if activity["status"] == "approved" and activity["result"]
    ]
    return jsonify({
        "totalProcessed": len(approved_activities),
        "reports": approved_activities
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)