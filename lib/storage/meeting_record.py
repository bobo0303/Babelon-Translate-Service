import datetime
from wjy3 import AbstractTable, CrudSql
from sqlalchemy.orm import Session
from sqlalchemy import Column, DateTime, Float, String, UnicodeText


class MeetingRecord(AbstractTable):
    _table_name = "Babelon_whisper_API"

    meeting_id = Column(String(32), nullable=False)
    device_id = Column(String(100), nullable=True)
    task_id = Column(String(100), nullable=True)
    audio_id = Column(String(32), nullable=False)
    audio_frame_timestamp = Column(DateTime, nullable=False)
    audio_file_name = Column(String(300), nullable=False)
    source_lang = Column(String(100), nullable=False)
    transcription_text = Column(UnicodeText, nullable=True)
    translation = Column(UnicodeText, nullable=False)
    transcribe_time = Column(Float, nullable=True)
    translate_time = Column(Float, nullable=True)
    audio_length = Column(Float, nullable=False)
    rtf = Column(Float, nullable=True)
    audio_tags = Column(String(100), nullable=False)
    strategy = Column(String(50), nullable=True)
    prev_text = Column(UnicodeText, nullable=True)
    
    # 保留原始 JSON 欄位作為備用 (存放不獨立的欄位)
    stt_data = Column(UnicodeText, nullable=False)

    def __init__(
        self,
        meeting_id: str,
        device_id: str = None,
        task_id: str = None,
        audio_id: str = None,
        audio_frame_timestamp: datetime = None,
        audio_file_name: str = None,
        source_lang: str = None,
        transcription_text: str = None,
        translation: str = None,
        transcribe_time: float = None,
        translate_time: float = None,
        audio_length: float = None,
        rtf: float = None,
        audio_tags: str = None,
        strategy: str = None,
        prev_text: str = None,
        stt_data: str = None,
    ):
        super().__init__()
        self.meeting_id = meeting_id
        self.device_id = device_id
        self.task_id = task_id
        self.audio_id = audio_id
        self.audio_frame_timestamp = audio_frame_timestamp
        self.audio_file_name = audio_file_name
        self.source_lang = source_lang
        self.transcription_text = transcription_text
        self.translation = translation
        self.transcribe_time = transcribe_time
        self.translate_time = translate_time
        self.audio_length = audio_length
        self.rtf = rtf
        self.audio_tags = audio_tags
        self.strategy = strategy
        self.prev_text = prev_text
        self.stt_data = stt_data


class MeetingRecordSql(CrudSql):
    def __init__(self, db: Session = None):
        super().__init__(MeetingRecord, db)

    def get_history_by_meeting(self, meeting_id: str):
        query = (
            self.db.query(
                MeetingRecord.uid.label("uid"),
                MeetingRecord.meeting_id.label("meeting_id"),
                MeetingRecord.speaker_id.label("speaker_id"),
                MeetingRecord.speaker_name.label("speaker_name"),
                MeetingRecord.source_lang.label("source_lang"),
                MeetingRecord.translation.label("translation"),
                MeetingRecord.audio_id.label("audio_id"),
                MeetingRecord.lm_time.label("lm_time"),
            )
            .filter(MeetingRecord.meeting_id == meeting_id)
            .order_by(MeetingRecord.lm_time.asc())
        )

        return query.all()
