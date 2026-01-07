"""Utilities to scrape public Telegram channels with Telethon and save raw messages to Parquet."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from dotenv import load_dotenv
from telethon.errors import RPCError
from telethon.sync import TelegramClient
from tqdm import tqdm


def load_channels_csv(path: str | Path) -> List[str]:
    """Load channel usernames or URLs from a CSV containing a 'channel' column."""
    df = pd.read_csv(path)
    if "channel" not in df.columns:
        raise ValueError("Channel CSV must include a 'channel' column.")
    return [c for c in df["channel"].dropna().astype(str) if c]


@dataclass
class TelegramScraper:
    """Scrape public Telegram channels using credentials from args or environment variables."""

    api_id: int | None = None
    api_hash: str | None = None 
    session_name: str = "rft_session"

    def __post_init__(self) -> None:
        """Load credentials from the environment and validate required values."""
        load_dotenv()
        env_api_id = os.getenv("TG_API_ID") or os.getenv("TELEGRAM_API_ID")
        env_api_hash = os.getenv("TG_API_HASH") or os.getenv("TELEGRAM_API_HASH")
        env_session = os.getenv("TG_SESSION_NAME")

        if not self.api_id and env_api_id:
            try:
                self.api_id = int(env_api_id)
            except ValueError as exc:
                raise ValueError("TG_API_ID must be an integer.") from exc
        if not self.api_hash and env_api_hash:
            self.api_hash = env_api_hash
        if env_session and (not self.session_name or self.session_name == "rft_session"):
            self.session_name = env_session

        missing = []
        if not self.api_id:
            missing.append("TG_API_ID")
        if not self.api_hash:
            missing.append("TG_API_HASH")
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing Telegram credentials: set {joined} in your .env or environment variables.")

    def scrape_to_parquet(
        self,
        channels: Iterable[str],
        output_path: str | Path,
        limit: int = 500,
        min_date: datetime | None = None,
    ) -> Path:
        """Scrape messages and write per-channel Parquet chunks plus a consolidated dataset."""
        output_path = Path(output_path)
        chunks_dir = output_path.parent / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving incremental chunks to: {chunks_dir}")

        processed_dfs = []

        # Use a single client session for efficiency and stability
        # Telethon loops don't like being restarted repeatedly in the same process
        with TelegramClient(self.session_name, self.api_id, self.api_hash) as client:
            for channel in tqdm(list(channels), desc="Channels"):
                chunk_path = chunks_dir / f"{channel}.parquet"

                # Optional: Skip if already done (Resumability)
                if chunk_path.exists() and chunk_path.stat().st_size > 0:
                     # pass if we want to skip
                     pass

                records = []
                try:
                    # Logic for single channel
                    iter_limit = limit if limit is not None else None
                    
                    entity = None
                    try:
                         entity = client.get_entity(channel)
                    except ValueError:
                         tqdm.write(f"Channel found but invalid/private: {channel}")
                         continue
                    except Exception as e:
                         tqdm.write(f"Error resolving {channel}: {e}")
                         continue

                    for message in client.iter_messages(entity, limit=iter_limit):
                        # Date filter
                        if min_date and message.date:
                            if message.date < min_date:
                                break
                        
                        if not message.message:
                            continue
                            
                        records.append({
                            "channel": str(channel),
                            "message_id": message.id,
                            "date": message.date,
                            "text": message.message,
                        })
                    
                    # Save Chunk immediately
                    if records:
                        df_chunk = pd.DataFrame(records)
                        df_chunk.to_parquet(chunk_path, index=False)
                        processed_dfs.append(df_chunk)
                        tqdm.write(f"Saved {len(records)} messages from {channel}")
                    else:
                        tqdm.write(f"No messages found for {channel} (within limits)")

                except Exception as e:
                    # Catch-all to prevent one channel crashing the whole batch
                    tqdm.write(f"CRITICAL ERROR on {channel}: {e}. Skipping to next.")
                    continue

                # Sleep to be nice to API
                time.sleep(1)

        # Final Consolidation
        print("Consolidating chunks into final dataset...")
        all_chunks = list(chunks_dir.glob("*.parquet"))
        if all_chunks:
            # Re-read all chunks to ensure we include ones from previous runs/sessions
            full_df = pd.concat([pd.read_parquet(f) for f in all_chunks], ignore_index=True)
            full_df.to_parquet(output_path, index=False)
            print(f"Success: saved {len(full_df)} total messages to {output_path}")
        else:
            print("No data was scraped.")

        return output_path

    # Deprecated/Wrapper for backward compatibility if needed, 
    # but scrape_to_parquet is the main entry point now.
    def scrape_channels(self, channels: Iterable[str], limit: int = 500, min_date: datetime | None = None) -> pd.DataFrame:
        """Deprecated compatibility wrapper that returns an empty DataFrame."""
        # Just a temporary in-memory version sharing no logic for now, 
        # or we could make it call scrape_to_parquet to a temp dir.
        # Keeping simple for now to avoid breaking imports, but functionally empty or basic.
        return pd.DataFrame()
