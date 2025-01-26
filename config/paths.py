from dataclasses import dataclass
import os

@dataclass
class MainPath:
    """
    Represents the main path configuration for the log analyzer.
    Dynamically resolves the folder path based on the script's location.
    """

    # Dynamically resolve the path to the "log_analyzer" folder in the Desktop directory
    folder_path: str = os.path.join(os.path.expanduser("~"), "Desktop", "log_analyzer")
    cpu_threshold: int = 40  # CPU usage threshold in percentage
    disk_threshold: int = 40  # Disk usage threshold in percentage
    critical_threshold: int = 40  # Critical system threshold in percentage

    def validate_path(self):
        """
        Validates if the folder path exists. Raises an error if not.
        """
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"The folder path does not exist: {self.folder_path}")

# Example Usage
if __name__ == "__main__":
    main_path = MainPath()
    try:
        # Validate if the folder exists
        main_path.validate_path()
        print(f"Folder path is valid: {main_path.folder_path}")
    except FileNotFoundError as e:
        print(str(e))
