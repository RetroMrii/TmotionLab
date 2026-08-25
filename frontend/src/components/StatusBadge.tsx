interface StatusBadgeProps {
    online: boolean;
}

export function StatusBadge({
    online,
}: StatusBadgeProps) {
    return (
        <span
            className={
                online
                    ? "status-badge status-online"
                    : "status-badge status-offline"
            }
        >
            {online ? "API online" : "API unavailable"}
        </span>
    );
}