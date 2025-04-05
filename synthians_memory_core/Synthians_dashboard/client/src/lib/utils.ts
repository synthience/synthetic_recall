import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import { formatDistanceToNow } from "date-fns"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Format a date as a relative time (e.g. "2 hours ago")
 */
export function formatTimeAgo(dateString: string | null | undefined): string {
  if (!dateString) return 'Unknown';
  try {
    const date = new Date(dateString);
    return formatDistanceToNow(date, { addSuffix: true });
  } catch (error) {
    console.error('Error formatting date:', dateString, error);
    return 'Invalid date';
  }
}
